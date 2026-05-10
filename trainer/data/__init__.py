"""trainer/data — Dataset management for TinyImageNet-200 and CIFAR-10."""
from trainer.data.tinyimagenet import (
    ensure_dataset,
    verify_dataset,
    make_train_loader,
    make_val_loader,
    DATASET_URL,
    DEFAULT_ROOT,
    NUM_CLASSES,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    IMG_SIZE,
    TRAIN_TRANSFORM,
    VAL_TRANSFORM,
)
from trainer.data.cifar10 import (
    ensure_cifar10,
    make_cifar10_train_loader,
    make_cifar10_val_loader,
    CIFAR10_ROOT,
    CIFAR10_NUM_CLASSES,
    CIFAR10_TRAIN_SAMPLES,
    CIFAR10_IMG_SIZE,
)

__all__ = [
    # TinyImageNet
    "ensure_dataset", "verify_dataset", "make_train_loader", "make_val_loader",
    "DATASET_URL", "DEFAULT_ROOT", "NUM_CLASSES", "TRAIN_SAMPLES",
    "VAL_SAMPLES", "IMG_SIZE", "TRAIN_TRANSFORM", "VAL_TRANSFORM",
    # CIFAR-10
    "ensure_cifar10", "make_cifar10_train_loader", "make_cifar10_val_loader",
    "CIFAR10_ROOT", "CIFAR10_NUM_CLASSES", "CIFAR10_TRAIN_SAMPLES", "CIFAR10_IMG_SIZE",
    # Unified
    "get_dataset_info", "ensure_any_dataset", "make_any_train_loader", "make_any_val_loader",
]


def get_dataset_info(dataset: str) -> dict:
    if dataset == "cifar10":
        return {
            "num_classes":   CIFAR10_NUM_CLASSES,
            "train_samples": CIFAR10_TRAIN_SAMPLES,
            "img_size":      CIFAR10_IMG_SIZE,
        }
    return {
        "num_classes":   NUM_CLASSES,
        "train_samples": TRAIN_SAMPLES,
        "img_size":      IMG_SIZE,
    }


def ensure_any_dataset(dataset: str, root=None) -> None:
    if dataset == "cifar10":
        ensure_cifar10(root)
    else:
        ensure_dataset(root)


def make_any_train_loader(dataset, root, indices, batch_size, cpu_cores):
    if dataset == "cifar10":
        return make_cifar10_train_loader(root, indices, batch_size, cpu_cores)
    return make_train_loader(root, indices, batch_size, cpu_cores)


def make_any_val_loader(dataset, root=None, batch_size=256, cpu_cores=4):
    if dataset == "cifar10":
        return make_cifar10_val_loader(root, batch_size, cpu_cores)
    return make_val_loader(root, batch_size, cpu_cores)
