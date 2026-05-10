"""trainer/data/cifar10 — CIFAR-10 download and DataLoader construction."""
from __future__ import annotations

from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T

CIFAR10_ROOT          = Path.home() / ".cache" / "cifar10"
CIFAR10_NUM_CLASSES   = 10
CIFAR10_TRAIN_SAMPLES = 50_000
CIFAR10_VAL_SAMPLES   = 10_000
CIFAR10_IMG_SIZE      = 32

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

CIFAR10_TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(CIFAR10_IMG_SIZE, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])

CIFAR10_VAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])


def ensure_cifar10(root: Path | None = None) -> Path:
    root = Path(root or CIFAR10_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(root=str(root), train=True,  download=True)
    datasets.CIFAR10(root=str(root), train=False, download=True)
    print(f"[dataset] CIFAR-10 ready at {root}")
    return root


def make_cifar10_train_loader(
    root:       Path | None,
    indices:    list[int],
    batch_size: int,
    cpu_cores:  int,
) -> torch.utils.data.DataLoader:
    root   = Path(root or CIFAR10_ROOT)
    full   = datasets.CIFAR10(root=str(root), train=True, transform=CIFAR10_TRAIN_TRANSFORM)
    subset = torch.utils.data.Subset(full, indices)
    nw     = min(4, max(1, cpu_cores // 2))
    return torch.utils.data.DataLoader(
        subset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = nw,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = nw > 0,
    )


def make_cifar10_val_loader(
    root:       Path | None = None,
    batch_size: int = 256,
    cpu_cores:  int = 4,
) -> torch.utils.data.DataLoader:
    root = Path(root or CIFAR10_ROOT)
    full = datasets.CIFAR10(root=str(root), train=False, transform=CIFAR10_VAL_TRANSFORM)
    nw   = min(4, max(1, cpu_cores // 2))
    return torch.utils.data.DataLoader(
        full,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = True,
    )
