"""
trainer/worker/synthetic_data.py — In-memory fake dataset for dry-run / testing.
No disk I/O; tensors look like real batches so the training loop runs unmodified.
"""
from __future__ import annotations

import torch
import torch.utils.data

from dataset import IMG_SIZE, NUM_CLASSES


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n: int, num_classes: int) -> None:
        self.n = n
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return (
            torch.randn(3, IMG_SIZE, IMG_SIZE),
            torch.randint(0, self.num_classes, (1,)).item(),
        )


def make_synthetic_loader(
    n: int, batch_size: int, num_classes: int
) -> torch.utils.data.DataLoader:
    ds = SyntheticDataset(n, num_classes)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
