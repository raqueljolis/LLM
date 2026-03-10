# MARVEL Multi-task Voice-based Detection (Paper Replica)

This folder contains a PyTorch implementation of the MARVEL architecture from  
**"Unified Multi-task Learning for Voice-Based Detection of Diverse Clinical Conditions"**.

### Files

- `marvel_model.py` – Implementation of the dual-branch multi-task model:
  - Spectrogram branch: EfficientNet-B0 on log-Mel spectrograms
  - MFCC branch: ResNet18 on MFCCs
  - Feature fusion: concatenation (1792-dim) → linear layer to 512-dim shared embedding
  - Task heads: 9 binary classification heads (512 → 128 → 1) with LeakyReLU, BatchNorm, Dropout
- `requirements.txt` – Minimal dependencies to install the model.
- `train_marvel.py` – Example training loop that mirrors the paper’s setup: AdamW, cosine LR schedule, balanced multi-task batch sampler, and gradient clipping.

### Install

From this folder:

```bash
pip install -r requirements.txt
```

### Usage Sketch

```python
import torch
from marvel_model import MarvelModel

# Create model
model = MarvelModel(num_tasks=9, pretrained=True)  # set pretrained=False if you prefer
model.train()

# Example batch of MFCC and spectrogram features
# Shapes follow the paper: (B, 1, T, F)
x_mfcc = torch.randn(32, 1, 256, 60)   # 60 MFCCs
x_spec = torch.randn(32, 1, 256, 128)  # 128 Mel filters

# Get logits for all 9 tasks
logits_all = model(x_mfcc, x_spec)  # (32, 9)

# Or train one specific task k with a 1D label vector
task_k = 0
logits_k = model(x_mfcc, x_spec, task_idx=task_k)  # (32,)
labels_k = torch.randint(0, 2, (32,), dtype=torch.float32)

from marvel_model import weighted_bce_loss

pos_weight = 2.0  # set from class frequencies per task
neg_weight = 1.0
loss = weighted_bce_loss(logits_k, labels_k, pos_weight, neg_weight)
loss.backward()
```

### Training loop (paper-style)

`train_marvel.py` contains:

- **Optimizer**: AdamW with initial learning rate `1e-4` and weight decay `1e-5`.
- **LR schedule**: cosine annealing (`CosineAnnealingLR`) over 40 epochs.
- **Batching**: batch size 108 with a balanced sampler:
  - 9 tasks × (6 positive + 6 negative) samples per task per batch.
  - Sampling with replacement when needed, as described in Section 4.1.
- **Loss**: per-task weighted BCE summed over all 9 tasks (Eq. 7).
- **Gradient clipping**: norm clipped to 1.0 each step.

Run it (after replacing the dummy data loader with your real MFCC/spectrogram tensors and labels):

```bash
python train_marvel.py
```

To fully match the paper, you should:

- Use precomputed MFCC and log-Mel spectrograms (e.g., from Bridge2AI-Voice v2.0).
- Implement a multi-task balanced batch sampler (6 positive + 6 negative samples per task per batch).
- Train with AdamW, cosine LR schedule, dropout 0.3, gradient clipping (norm 1.0), and data augmentations as described in Section 4.1 of the paper.

