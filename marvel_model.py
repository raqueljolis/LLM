import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MarvelModel(nn.Module):
    """
    PyTorch implementation of the MARVEL multi-task dual-branch architecture from
    "Unified Multi-task Learning for Voice-Based Detection of Diverse Clinical Conditions".

    - Spectrogram branch: EfficientNet-B0 on log-Mel spectrograms
    - MFCC branch: ResNet18 on MFCCs
    - Fused shared representation: 1792 -> 512
    - Task-specific heads: 512 -> 128 -> 1 (per task), with LeakyReLU, BatchNorm, Dropout
    """

    def __init__(
        self,
        num_tasks: int = 9,
        shared_dim: int = 512,
        head_hidden_dim: int = 128,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------------------
        # Spectrogram encoder: EfficientNet-B0 (log-Mel spectrograms)
        # ---------------------------------------------------------------------
        effnet_weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        effnet = models.efficientnet_b0(weights=effnet_weights)

        # Project single-channel spectrogram (1 x T x F) to 3 channels
        self.spec_preconv = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
        self.spec_preconv_bn = nn.BatchNorm2d(3)

        # Use EfficientNet feature extractor and its global average pooling
        self.spec_backbone = effnet.features
        self.spec_pool = effnet.avgpool
        spec_out_dim = effnet.classifier[1].in_features  # 1280 for EfficientNet-B0

        # ---------------------------------------------------------------------
        # MFCC encoder: ResNet18 (MFCCs)
        # ---------------------------------------------------------------------
        resnet_weights = (
            models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        resnet = models.resnet18(weights=resnet_weights)

        # Modify first conv layer to accept single-channel MFCC input
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=False,
        )

        self.mfcc_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.mfcc_pool = resnet.avgpool
        mfcc_out_dim = resnet.fc.in_features  # 512 for ResNet18

        # ---------------------------------------------------------------------
        # Shared fusion layer: concat embeddings then project to shared_dim (512)
        # ---------------------------------------------------------------------
        fused_dim = spec_out_dim + mfcc_out_dim  # 1280 + 512 = 1792
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        # ---------------------------------------------------------------------
        # Task-specific heads: one 2-layer MLP per diagnostic task
        # ---------------------------------------------------------------------
        heads = {}
        for k in range(num_tasks):
            heads[str(k)] = nn.Sequential(
                nn.Linear(shared_dim, head_hidden_dim),
                nn.BatchNorm1d(head_hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        self.heads = nn.ModuleDict(heads)
        self.num_tasks = num_tasks

    def encode(self, x_mfcc: torch.Tensor, x_spec: torch.Tensor) -> torch.Tensor:
        """
        Run inputs through both modality-specific encoders and return fused shared embedding z.

        Args:
            x_mfcc: MFCC tensor of shape (B, 1, T_mfcc, F_mfcc) or (B, 1, H, W)
            x_spec: log-Mel spectrogram tensor of shape (B, 1, T_spec, F_spec) or (B, 1, H, W)

        Returns:
            z: shared embedding of shape (B, shared_dim)
        """
        # MFCC branch
        h_mfcc = self.mfcc_backbone(x_mfcc)
        h_mfcc = self.mfcc_pool(h_mfcc)  # (B, 512, 1, 1)
        h_mfcc = torch.flatten(h_mfcc, 1)  # (B, 512)

        # Spectrogram branch
        x_spec = self.spec_preconv(x_spec)
        x_spec = self.spec_preconv_bn(x_spec)
        h_spec = self.spec_backbone(x_spec)
        h_spec = self.spec_pool(h_spec)  # (B, 1280, 1, 1)
        h_spec = torch.flatten(h_spec, 1)  # (B, 1280)

        # Fuse and project to shared space
        h_fused = torch.cat([h_mfcc, h_spec], dim=1)  # (B, 1792)
        z = self.shared(h_fused)  # (B, shared_dim)
        return z

    def forward(
        self,
        x_mfcc: torch.Tensor,
        x_spec: torch.Tensor,
        task_idx: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through MARVEL.

        Args:
            x_mfcc: MFCC input, shape (B, 1, H, W)
            x_spec: Spectrogram input, shape (B, 1, H, W)
            task_idx:
                - None: return logits for all tasks, shape (B, num_tasks)
                - int: return logits for a single task, shape (B,)
                - LongTensor of shape (B,): per-sample task index, return logits (B,)

        Returns:
            Logits (before sigmoid). Apply torch.sigmoid() externally for probabilities.
        """
        z = self.encode(x_mfcc, x_spec)

        # Case 1: return logits for all tasks (B, K)
        if task_idx is None:
            logits_per_head = []
            for k in range(self.num_tasks):
                head = self.heads[str(k)]
                logits_per_head.append(head(z))  # (B, 1)
            logits = torch.cat(logits_per_head, dim=1)  # (B, K)
            return logits

        # Case 2: single scalar task index for whole batch
        if isinstance(task_idx, int):
            head = self.heads[str(task_idx)]
            logits = head(z).squeeze(-1)  # (B,)
            return logits

        # Case 3: per-sample task indices (B,)
        if not torch.is_tensor(task_idx):
            raise TypeError("task_idx must be None, int, or LongTensor.")

        if task_idx.dim() != 1 or task_idx.size(0) != z.size(0):
            raise ValueError(
                "task_idx tensor must have shape (batch_size,) matching inputs."
            )

        device = z.device
        batch_size = z.size(0)
        logits = torch.empty(batch_size, device=device)

        for k_str, head in self.heads.items():
            k = int(k_str)
            mask = task_idx == k
            if mask.any():
                logits[mask] = head(z[mask]).squeeze(-1)

        return logits


class MarvelSSLAutoencoder(nn.Module):
    """
    SSL autoencoder variant using the same dual-branch encoder idea:
      - MFCC branch: ResNet18
      - Spectrogram branch: EfficientNet-B0

    The fused latent embedding is decoded to reconstruct both modalities.
    """

    def __init__(
        self,
        shared_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------------------
        # Spectrogram encoder branch (EfficientNet-B0)
        # ---------------------------------------------------------------------
        effnet_weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        effnet = models.efficientnet_b0(weights=effnet_weights)
        self.spec_preconv = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
        self.spec_preconv_bn = nn.BatchNorm2d(3)
        self.spec_backbone = effnet.features
        self.spec_pool = effnet.avgpool
        spec_out_dim = effnet.classifier[1].in_features  # 1280

        # ---------------------------------------------------------------------
        # MFCC encoder branch (ResNet18)
        # ---------------------------------------------------------------------
        resnet_weights = (
            models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        resnet = models.resnet18(weights=resnet_weights)
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=False,
        )
        self.mfcc_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.mfcc_pool = resnet.avgpool
        mfcc_out_dim = resnet.fc.in_features  # 512

        # ---------------------------------------------------------------------
        # Shared latent projection
        # ---------------------------------------------------------------------
        fused_dim = spec_out_dim + mfcc_out_dim  # 1792
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.LeakyReLU(0.1),
        )

        # ---------------------------------------------------------------------
        # Lightweight decoders from latent z to each modality
        # ---------------------------------------------------------------------
        self.mfcc_decoder_fc = nn.Linear(shared_dim, 64 * 16 * 16)
        self.mfcc_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        self.spec_decoder_fc = nn.Linear(shared_dim, 64 * 16 * 16)
        self.spec_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def encode(self, x_mfcc: torch.Tensor, x_spec: torch.Tensor) -> torch.Tensor:
        # MFCC branch
        h_mfcc = self.mfcc_backbone(x_mfcc)
        h_mfcc = self.mfcc_pool(h_mfcc)
        h_mfcc = torch.flatten(h_mfcc, 1)

        # Spectrogram branch
        x_spec_3ch = self.spec_preconv(x_spec)
        x_spec_3ch = self.spec_preconv_bn(x_spec_3ch)
        h_spec = self.spec_backbone(x_spec_3ch)
        h_spec = self.spec_pool(h_spec)
        h_spec = torch.flatten(h_spec, 1)

        # Fuse
        h_fused = torch.cat([h_mfcc, h_spec], dim=1)
        z = self.shared(h_fused)
        return z

    def decode(
        self,
        z: torch.Tensor,
        mfcc_shape: torch.Size,
        spec_shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # MFCC reconstruction
        r_mfcc = self.mfcc_decoder_fc(z).view(z.size(0), 64, 16, 16)
        r_mfcc = self.mfcc_decoder(r_mfcc)
        r_mfcc = F.interpolate(
            r_mfcc,
            size=(mfcc_shape[-2], mfcc_shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        # Spectrogram reconstruction
        r_spec = self.spec_decoder_fc(z).view(z.size(0), 64, 16, 16)
        r_spec = self.spec_decoder(r_spec)
        r_spec = F.interpolate(
            r_spec,
            size=(spec_shape[-2], spec_shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return r_mfcc, r_spec

    def forward(
        self,
        x_mfcc: torch.Tensor,
        x_spec: torch.Tensor,
        return_embedding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x_mfcc, x_spec)
        recon_mfcc, recon_spec = self.decode(z, x_mfcc.shape, x_spec.shape)
        if return_embedding:
            return recon_mfcc, recon_spec, z
        return recon_mfcc, recon_spec

    @staticmethod
    def reconstruction_loss(
        x_mfcc: torch.Tensor,
        x_spec: torch.Tensor,
        recon_mfcc: torch.Tensor,
        recon_spec: torch.Tensor,
        mfcc_weight: float = 1.0,
        spec_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Weighted MSE reconstruction loss for SSL training.
        """
        loss_mfcc = F.mse_loss(recon_mfcc, x_mfcc)
        loss_spec = F.mse_loss(recon_spec, x_spec)
        return mfcc_weight * loss_mfcc + spec_weight * loss_spec


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    neg_weight: float,
) -> torch.Tensor:
    """
    Weighted binary cross-entropy loss as described in the paper.

    Args:
        logits: model outputs before sigmoid, shape (N,) or (N, 1)
        targets: binary labels in {0, 1}, same shape as logits
        pos_weight: weight for positive class (w1)
        neg_weight: weight for negative class (w0)
    """
    logits = logits.view(-1)
    targets = targets.view(-1)

    probs = torch.sigmoid(logits)
    loss_pos = -pos_weight * targets * torch.log(probs.clamp_min(1e-8))
    loss_neg = -neg_weight * (1.0 - targets) * torch.log(
        (1.0 - probs).clamp_min(1e-8)
    )
    return (loss_pos + loss_neg).mean()


if __name__ == "__main__":
    # Minimal sanity check with dummy inputs
    batch_size = 4
    # Example shapes: (B, 1, T, F)
    x_mfcc = torch.randn(batch_size, 1, 256, 60)
    x_spec = torch.randn(batch_size, 1, 256, 128)

    model = MarvelModel(num_tasks=9, pretrained=False)
    with torch.no_grad():
        # All-task logits
        logits_all = model(x_mfcc, x_spec)  # (B, 9)
        print("All-task logits shape:", logits_all.shape)

        # Single-task logits
        logits_task_0 = model(x_mfcc, x_spec, task_idx=0)  # (B,)
        print("Task 0 logits shape:", logits_task_0.shape)

        # Per-sample task indices
        task_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        logits_per_sample = model(x_mfcc, x_spec, task_idx=task_indices)
        print("Per-sample logits shape:", logits_per_sample.shape)

        # SSL autoencoder sanity check
        ssl_model = MarvelSSLAutoencoder(pretrained=False)
        recon_mfcc, recon_spec, z = ssl_model(x_mfcc, x_spec, return_embedding=True)
        print("Recon MFCC shape:", recon_mfcc.shape)
        print("Recon Spec shape:", recon_spec.shape)
        print("Embedding shape:", z.shape)
        print(
            "Reconstruction loss:",
            MarvelSSLAutoencoder.reconstruction_loss(
                x_mfcc, x_spec, recon_mfcc, recon_spec
            ).item(),
        )

