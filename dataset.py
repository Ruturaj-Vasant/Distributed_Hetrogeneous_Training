"""
dataset.py  —  Dataset management for Tiny ImageNet-200 and CIFAR-10

Handles download, setup, and DataLoader construction for both datasets.

Can be run standalone to pre-download:
    python3 dataset.py [--dataset cifar10|tinyimagenet] [--root <path>]
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T

# ── Tiny ImageNet-200 constants ───────────────────────────────────────────────

DATASET_URL   = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_ROOT  = Path.home() / ".cache" / "tiny-imagenet-200"
NUM_CLASSES   = 200
TRAIN_SAMPLES = 100_000
VAL_SAMPLES   = 10_000
IMG_SIZE      = 64

# ── CIFAR-10 constants ────────────────────────────────────────────────────────

CIFAR10_ROOT          = Path.home() / ".cache" / "cifar10"
CIFAR10_NUM_CLASSES   = 10
CIFAR10_TRAIN_SAMPLES = 50_000
CIFAR10_VAL_SAMPLES   = 10_000
CIFAR10_IMG_SIZE      = 32

# ── Transforms ────────────────────────────────────────────────────────────────

_MEAN = [0.4802, 0.4481, 0.3975]
_STD  = [0.2770, 0.2691, 0.2821]

TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

VAL_TRANSFORM = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])


# ── Download & setup ──────────────────────────────────────────────────────────

def ensure_dataset(root: Path | None = None) -> Path:
    """
    Ensure Tiny ImageNet-200 is present at `root`.
    Returns the path to the `train/` directory.

    Steps:
      1. Download the zip if not cached
      2. Extract
      3. Reorganise val/ from flat layout to ImageFolder layout
    """
    root = Path(root or DEFAULT_ROOT)
    train_dir = root / "tiny-imagenet-200" / "train"
    val_dir   = root / "tiny-imagenet-200" / "val"

    if _dataset_ready(train_dir, val_dir):
        print(f"[dataset] Found at {root}")
        return train_dir

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        _download(DATASET_URL, zip_path)

    print("[dataset] Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    _reorganise_val(val_dir)
    print(f"[dataset] Ready at {root}")
    return train_dir


def _dataset_ready(train_dir: Path, val_dir: Path) -> bool:
    if not train_dir.exists() or not val_dir.exists():
        return False
    # Val is ready when it contains class subdirs (not the raw flat layout)
    val_subdirs = [d for d in val_dir.iterdir() if d.is_dir() and d.name != "images"]
    return len(val_subdirs) == NUM_CLASSES


def _download(url: str, dest: Path) -> None:
    print(f"[dataset] Downloading Tiny ImageNet-200 (~236 MB) from {url} …")

    def _hook(count, block, total):
        if total > 0:
            pct = min(100.0, count * block / total * 100)
            bar = "#" * int(pct // 2)
            print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)
    print()


def _reorganise_val(val_dir: Path) -> None:
    """
    The raw val/ layout is flat:
        val/images/val_0.JPEG
        val/val_annotations.txt  (filename TAB classname TAB ...)

    ImageFolder expects:
        val/<classname>/val_0.JPEG

    This function does the rename in-place and removes val/images/ afterwards.
    Safe to call multiple times (skips already-moved files).
    """
    ann_file = val_dir / "val_annotations.txt"
    img_dir  = val_dir / "images"
    if not ann_file.exists() or not img_dir.exists():
        return

    print("[dataset] Reorganising val/ into ImageFolder layout …")
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            fname, cls = parts[0], parts[1]
            cls_dir = val_dir / cls
            cls_dir.mkdir(exist_ok=True)
            src = img_dir / fname
            dst = cls_dir / fname
            if src.exists():
                shutil.move(str(src), dst)

    # Remove the now-empty flat images dir
    try:
        img_dir.rmdir()
    except OSError:
        pass


def verify_dataset(root: Path | None = None) -> bool:
    """Return True if the dataset looks complete (correct class/image counts)."""
    root      = Path(root or DEFAULT_ROOT)
    train_dir = root / "tiny-imagenet-200" / "train"
    val_dir   = root / "tiny-imagenet-200" / "val"

    if not _dataset_ready(train_dir, val_dir):
        return False

    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes   = [d for d in val_dir.iterdir()   if d.is_dir()]

    if len(train_classes) != NUM_CLASSES or len(val_classes) != NUM_CLASSES:
        return False

    # Spot-check: each train class should have 500 images
    sample_cls   = train_classes[0]
    sample_imgs  = list((sample_cls / "images").glob("*.JPEG"))
    return len(sample_imgs) == 500


# ── DataLoader factories ──────────────────────────────────────────────────────

def make_train_loader(
    root:        Path | None,
    indices:     list[int],
    batch_size:  int,
    cpu_cores:   int,
) -> torch.utils.data.DataLoader:
    """
    DataLoader for the worker's assigned shard of the training split.
    `indices` is the list of global sample indices this worker owns.
    """
    train_dir = Path(root or DEFAULT_ROOT) / "tiny-imagenet-200" / "train"
    full      = datasets.ImageFolder(str(train_dir), transform=TRAIN_TRANSFORM)
    subset    = torch.utils.data.Subset(full, indices)
    nw        = min(4, max(1, cpu_cores // 2))
    return torch.utils.data.DataLoader(
        subset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = nw,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = nw > 0,
    )


def make_val_loader(
    root:       Path | None,
    batch_size: int = 128,
    cpu_cores:  int = 4,
) -> torch.utils.data.DataLoader:
    """DataLoader for the full validation split (all 10,000 images)."""
    val_dir = Path(root or DEFAULT_ROOT) / "tiny-imagenet-200" / "val"
    full    = datasets.ImageFolder(str(val_dir), transform=VAL_TRANSFORM)
    nw      = min(4, max(1, cpu_cores // 2))
    return torch.utils.data.DataLoader(
        full,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = True,
    )


# ── CIFAR-10 support ──────────────────────────────────────────────────────────

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
    """Download CIFAR-10 via torchvision if not already cached."""
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


# ── Unified interface (used by leader and worker) ──────────────────────────────

def get_dataset_info(dataset: str) -> dict:
    """Return constants for a given dataset name."""
    if dataset == "cifar10":
        return {
            "num_classes":    CIFAR10_NUM_CLASSES,
            "train_samples":  CIFAR10_TRAIN_SAMPLES,
            "img_size":       CIFAR10_IMG_SIZE,
        }
    return {
        "num_classes":    NUM_CLASSES,
        "train_samples":  TRAIN_SAMPLES,
        "img_size":       IMG_SIZE,
    }


def ensure_any_dataset(dataset: str, root: Path | None = None) -> None:
    """Download the named dataset if not already cached."""
    if dataset == "cifar10":
        ensure_cifar10(root)
    else:
        ensure_dataset(root)


def make_any_train_loader(
    dataset:    str,
    root:       Path | None,
    indices:    list[int],
    batch_size: int,
    cpu_cores:  int,
) -> torch.utils.data.DataLoader:
    if dataset == "cifar10":
        return make_cifar10_train_loader(root, indices, batch_size, cpu_cores)
    return make_train_loader(root, indices, batch_size, cpu_cores)


def make_any_val_loader(
    dataset:    str,
    root:       Path | None = None,
    batch_size: int = 256,
    cpu_cores:  int = 4,
) -> torch.utils.data.DataLoader:
    if dataset == "cifar10":
        return make_cifar10_val_loader(root, batch_size, cpu_cores)
    return make_val_loader(root, batch_size, cpu_cores)


# ── Standalone download ───────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download and prepare a training dataset")
    p.add_argument("--dataset", default="tinyimagenet", choices=["tinyimagenet", "cifar10"])
    p.add_argument("--root", default=None, help="Cache directory override")
    args = p.parse_args()

    root = Path(args.root) if args.root else None
    ensure_any_dataset(args.dataset, root)

    if args.dataset == "tinyimagenet":
        ok = verify_dataset(root)
        print(f"[dataset] Integrity check: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("[dataset] WARNING: dataset may be incomplete.")
