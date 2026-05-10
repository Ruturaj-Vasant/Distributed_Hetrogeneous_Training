"""trainer/data/tinyimagenet — TinyImageNet-200 download, setup, and DataLoader construction."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T

DATASET_URL   = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_ROOT  = Path.home() / ".cache" / "tiny-imagenet-200"
NUM_CLASSES   = 200
TRAIN_SAMPLES = 100_000
VAL_SAMPLES   = 10_000
IMG_SIZE      = 64

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


def ensure_dataset(root: Path | None = None) -> Path:
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

    try:
        img_dir.rmdir()
    except OSError:
        pass


def verify_dataset(root: Path | None = None) -> bool:
    root      = Path(root or DEFAULT_ROOT)
    train_dir = root / "tiny-imagenet-200" / "train"
    val_dir   = root / "tiny-imagenet-200" / "val"

    if not _dataset_ready(train_dir, val_dir):
        return False

    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes   = [d for d in val_dir.iterdir()   if d.is_dir()]

    if len(train_classes) != NUM_CLASSES or len(val_classes) != NUM_CLASSES:
        return False

    sample_cls  = train_classes[0]
    sample_imgs = list((sample_cls / "images").glob("*.JPEG"))
    return len(sample_imgs) == 500


def make_train_loader(
    root:       Path | None,
    indices:    list[int],
    batch_size: int,
    cpu_cores:  int,
) -> torch.utils.data.DataLoader:
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
