import math
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from marvel_model import MarvelModel, weighted_bce_loss


class MultiTaskVoiceDataset(Dataset):
    """
    Thin wrapper you can adapt to your data.

    Expects __getitem__ to return:
        x_mfcc:  (1, T_mfcc, F_mfcc)
        x_spec:  (1, T_spec, F_spec)
        labels:  (K,) tensor with 0/1 for K tasks
    """

    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_mfcc, x_spec, labels = self.samples[idx]
        return x_mfcc, x_spec, labels


class BalancedMultiTaskBatchSampler(Sampler[List[int]]):
    """
    Implements the paper's balanced batch sampler:
    - Batch size 108
    - 9 tasks
    - 12 samples per task per batch: 6 positive, 6 negative
    Sampling is done with replacement when needed.
    """

    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int = 108,
        num_tasks: int = 9,
        positives_per_task: int = 6,
        negatives_per_task: int = 6,
        max_batches_per_epoch: int | None = None,
    ) -> None:
        """
        Args:
            labels: tensor of shape (N, K) with 0/1 labels for K tasks.
            batch_size: should be num_tasks * (positives_per_task + negatives_per_task)
            num_tasks: K (9 in the paper)
            positives_per_task: number of positive samples per task in each batch
            negatives_per_task: number of negative samples per task in each batch
            max_batches_per_epoch: optional limit for steps per epoch
        """
        if labels.dim() != 2:
            raise ValueError("labels must be of shape (N, K)")

        self.labels = labels
        self.num_tasks = num_tasks
        self.positives_per_task = positives_per_task
        self.negatives_per_task = negatives_per_task
        self.batch_size = batch_size
        self.max_batches_per_epoch = max_batches_per_epoch

        expected = num_tasks * (positives_per_task + negatives_per_task)
        if batch_size != expected:
            raise ValueError(
                f"batch_size must be K * (pos + neg) = {expected}, got {batch_size}"
            )

        # Pre-compute indices per task for positives and negatives
        self.pos_indices_per_task: Dict[int, torch.Tensor] = {}
        self.neg_indices_per_task: Dict[int, torch.Tensor] = {}
        for k in range(num_tasks):
            task_labels = labels[:, k]
            pos_idx = torch.nonzero(task_labels == 1, as_tuple=False).view(-1)
            neg_idx = torch.nonzero(task_labels == 0, as_tuple=False).view(-1)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                raise ValueError(
                    f"Task {k} has no positive or no negative samples; cannot build balanced batches."
                )
            self.pos_indices_per_task[k] = pos_idx
            self.neg_indices_per_task[k] = neg_idx

    def __iter__(self):
        num_samples = self.labels.size(0)

        # Rough upper bound on number of unique batches possible
        if self.max_batches_per_epoch is not None:
            num_batches = self.max_batches_per_epoch
        else:
            # Heuristic: every sample used about once per epoch on average
            num_batches = max(1, math.ceil(num_samples / self.batch_size))

        rng = torch.Generator()
        for _ in range(num_batches):
            batch_indices: List[int] = []
            for k in range(self.num_tasks):
                pos_pool = self.pos_indices_per_task[k]
                neg_pool = self.neg_indices_per_task[k]

                pos_choice = pos_pool[
                    torch.randint(
                        high=len(pos_pool),
                        size=(self.positives_per_task,),
                        generator=rng,
                    )
                ]
                neg_choice = neg_pool[
                    torch.randint(
                        high=len(neg_pool),
                        size=(self.negatives_per_task,),
                        generator=rng,
                    )
                ]
                batch_indices.extend(pos_choice.tolist())
                batch_indices.extend(neg_choice.tolist())

            yield batch_indices

    def __len__(self) -> int:
        if self.max_batches_per_epoch is not None:
            return self.max_batches_per_epoch
        num_samples = self.labels.size(0)
        return max(1, math.ceil(num_samples / self.batch_size))


def compute_class_weights(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-task positive/negative weights inverse to class frequencies.

    Args:
        labels: (N, K) tensor with 0/1.
    Returns:
        pos_weights: (K,)
        neg_weights: (K,)
    """
    N, K = labels.shape
    pos_counts = labels.sum(dim=0).float()
    neg_counts = (N - pos_counts).float()

    # Avoid division by zero
    pos_weights = (neg_counts / (pos_counts + 1e-8)).clamp_min(1.0)
    neg_weights = (pos_counts / (neg_counts + 1e-8)).clamp_min(1.0)
    return pos_weights, neg_weights


def train_one_epoch(
    model: MarvelModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    pos_weights: torch.Tensor,
    neg_weights: torch.Tensor,
    grad_clip_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x_mfcc, x_spec, labels in dataloader:
        x_mfcc = x_mfcc.to(device)
        x_spec = x_spec.to(device)
        labels = labels.to(device).float()  # (B, K)

        optimizer.zero_grad(set_to_none=True)

        logits_all = model(x_mfcc, x_spec)  # (B, K)

        # Sum of weighted BCE over tasks (Eq. 7 in paper)
        batch_loss = 0.0
        K = logits_all.size(1)
        for k in range(K):
            logits_k = logits_all[:, k]
            labels_k = labels[:, k]
            w1_k = pos_weights[k].item()
            w0_k = neg_weights[k].item()
            batch_loss = batch_loss + weighted_bce_loss(
                logits_k, labels_k, pos_weight=w1_k, neg_weight=w0_k
            )

        batch_loss.backward()

        # Gradient clipping (norm 1.0 as in paper)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        total_loss += batch_loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    # -------------------------------------------------------------------------
    # Configuration (matches paper defaults)
    # -------------------------------------------------------------------------
    num_tasks = 9
    batch_size = 108
    max_epochs = 40
    initial_lr = 1e-4
    weight_decay = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # TODO: Replace this with real data loading (Bridge2AI-Voice or your own).
    # For now we create dummy tensors to illustrate the pipeline.
    # -------------------------------------------------------------------------
    num_samples = 1000
    T_mfcc, F_mfcc = 256, 60
    T_spec, F_spec = 256, 128

    dummy_samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(num_samples):
        x_mfcc = torch.randn(1, T_mfcc, F_mfcc)
        x_spec = torch.randn(1, T_spec, F_spec)
        labels = torch.randint(0, 2, (num_tasks,), dtype=torch.long)
        dummy_samples.append((x_mfcc, x_spec, labels))

    dataset = MultiTaskVoiceDataset(dummy_samples)

    # Build labels matrix for sampler / class weights
    all_labels = torch.stack([s[2] for s in dummy_samples], dim=0)  # (N, K)

    sampler = BalancedMultiTaskBatchSampler(
        labels=all_labels,
        batch_size=batch_size,
        num_tasks=num_tasks,
        positives_per_task=6,
        negatives_per_task=6,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
    )

    # Compute per-task class weights
    pos_weights, neg_weights = compute_class_weights(all_labels)

    # -------------------------------------------------------------------------
    # Model, optimizer, LR scheduler, gradient clipping
    # -------------------------------------------------------------------------
    model = MarvelModel(num_tasks=num_tasks, pretrained=True)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
    )

    # Cosine annealing schedule over max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            pos_weights=pos_weights,
            neg_weights=neg_weights,
            grad_clip_norm=1.0,
        )

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{max_epochs} - "
            f"loss: {train_loss:.4f} - lr: {current_lr:.6f}"
        )


if __name__ == "__main__":
    main()

