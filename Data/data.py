"""Week 1 - Data loading and preprocessing for CIFAR-10.

Complete every block marked TODO. Read README.md of Week 1 first: it explains
the background (tensors, normalization, one-hot encoding, splits, augmentation).

Work in this order (each step can be tested on its own, see README "Step-by-step checks"):
    1. compute_mean_std      5. load_cifar10                    9.  benchmark_loader
    2. build_transforms      6. make_loaders                    10. describe
    3. one_hot / OneHot      7. class_counts / count_overlap    11. save_augmentation_grid
    4. split_indices         8. channel_stats

Useful documentation:
    torchvision transforms: https://pytorch.org/vision/stable/transforms.html
    Dataset / DataLoader:   https://pytorch.org/docs/stable/data.html
"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
NUM_CLASSES = len(CLASSES)
AUGMENTATIONS = ("none", "flip", "crop_flip", "strong")


def compute_mean_std(images):
    """Per-channel mean/std of uint8 images shaped (N, H, W, 3), after scaling to [0, 1].

    Returns two tuples of 3 Python floats: (mean_R, mean_G, mean_B), (std_R, std_G, std_B).

    Example (whole CIFAR-10 training set):
        mean ~ (0.491, 0.482, 0.447), std ~ (0.247, 0.243, 0.262)
    """
    # TODO 1: compute the per-channel statistics.
    #   a) Convert `images` to float32 and divide by 255.0 so values are in [0, 1].
    #   b) The array has shape (N, H, W, C). Reduce over every axis EXCEPT the channel axis
    #      (hint: axis=(0, 1, 2)) with .mean(...) and .std(...).
    #   c) Return them as tuples of Python floats: tuple(array.tolist()).
    images = images.astype(np.float32) / 255.0
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))
    return tuple(mean.tolist()), tuple(std.tolist())


def build_transforms(normalize=True, augment="crop_flip", mean=None, std=None):
    """Return (train_transform, eval_transform).

    augment:   "none" | "flip" | "crop_flip" | "strong"  (training only)
    normalize: subtract mean / divide by std per channel (train and eval).

    What each augment level contains:
        "none"      -> nothing
        "flip"      -> RandomHorizontalFlip
        "crop_flip" -> RandomCrop(32, padding=4) + RandomHorizontalFlip
        "strong"    -> RandomCrop + RandomHorizontalFlip + ColorJitter(0.3, 0.3, 0.3)
                       and, after ToTensor/Normalize, RandomErasing(p=0.5, scale=(0.02, 0.15))
    """
    if augment not in AUGMENTATIONS:
        raise ValueError(f"Unknown augment: {augment}. Choose from {AUGMENTATIONS}")

    # TODO 2a: build the list `pil_aug` of augmentations that work on PIL images
    #   (RandomCrop, RandomHorizontalFlip, ColorJitter) according to the table above.
    #   Hint: start with pil_aug = [] and use `if augment in (...)` checks to append.
    pil_aug = []
    if augment in ("crop_flip", "strong"):
        pil_aug.append(transforms.RandomCrop(32, padding=4))
    if augment in ("flip", "crop_flip", "strong"):
        pil_aug.append(transforms.RandomHorizontalFlip())
    if augment == "strong":
        pil_aug.append(transforms.ColorJitter(0.3, 0.3, 0.3))

    # TODO 2b: build the list `base` that BOTH train and eval use:
    #   always transforms.ToTensor()  (PIL uint8 HxWxC [0,255] -> float CxHxW [0,1]),
    #   plus transforms.Normalize(mean, std) if normalize is True.
    base = [transforms.ToTensor()]
    if normalize:
        base.append(transforms.Normalize(mean, std))

    # TODO 2c: build `tensor_aug`: [RandomErasing(...)] only for "strong", otherwise [].
    #   Question for you: why must RandomErasing come AFTER ToTensor?
    tensor_aug = []
    if augment == "strong":
        tensor_aug.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)))

    # TODO 2d: return transforms.Compose(pil_aug + base + tensor_aug), transforms.Compose(base)
    #   Question for you: why does the eval transform contain NO augmentation?
    return transforms.Compose(pil_aug + base + tensor_aug), transforms.Compose(base)


def one_hot(label, num_classes=NUM_CLASSES, smoothing=0.0):
    """Integer label -> float one-hot vector, optionally label-smoothed.

    smoothing=0.0 gives a hard one-hot [0, ..., 1, ..., 0];
    smoothing=0.1 gives 0.91 on the true class and 0.01 elsewhere.
    """
    # TODO 3a: create a zero tensor of length num_classes, set position `label` to 1.0.
    # TODO 3b: apply label smoothing:  target * (1 - smoothing) + smoothing / num_classes
    #   Check: the result must always sum to 1.0. Can you prove it on paper?
    target = torch.zeros(num_classes, dtype=torch.float32)
    target[int(label)] = 1.0
    if smoothing > 0.0:
        target = target * (1.0 - smoothing) + smoothing / num_classes
    return target


class OneHot:
    """Picklable target_transform so it works with DataLoader workers.

    (A lambda would not work with num_workers > 0, because lambdas cannot be pickled
    and sent to the worker processes.)
    """

    def __init__(self, num_classes=NUM_CLASSES, smoothing=0.0):
        self.num_classes = num_classes
        self.smoothing = smoothing

    def __call__(self, label):
        # TODO 3c: call one_hot(...) with the stored settings.
        return one_hot(label, self.num_classes, self.smoothing)


def split_indices(n, val_split=0.1, subset_fraction=1.0, seed=42, leak="none"):
    """Shuffle 0..n-1, keep subset_fraction of them, and cut off val_split for validation.

    Returns (train_idx, val_idx) as numpy integer arrays.

    leak="val_in_train" deliberately puts the validation images into the training set too
    (a data-leakage bug, used for the Week 3 experiments).

    Example: n=50000, val_split=0.1, subset_fraction=0.2 -> 9000 train, 1000 val.
    """
    # TODO 4:
    #   a) rng = np.random.default_rng(seed); indices = rng.permutation(n)
    #      (a fixed seed means everyone gets the SAME split every run - reproducibility!)
    #   b) keep only the first int(n * subset_fraction) indices
    #   c) n_val = int(len(indices) * val_split); the first n_val go to validation,
    #      the rest to training
    #   d) if leak == "val_in_train": make train_idx contain ALL kept indices (bug on purpose)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    kept = indices[: int(n * subset_fraction)]
    n_val = int(len(kept) * val_split)
    val_idx = kept[:n_val]
    train_idx = kept[n_val:]
    if leak == "val_in_train":
        train_idx = kept
    return train_idx, val_idx


def load_cifar10(data_dir="./data", val_split=0.1, subset_fraction=1.0, normalize=True,
                 augment="crop_flip", label_smoothing=0.0, seed=42, leak="none"):
    """Download CIFAR-10 and return (train_set, val_set, test_set).

    Labels are one-hot encoded. Label smoothing is only applied to the training set;
    validation/test keep hard one-hot targets. Normalization statistics are computed
    on the training split only.
    """
    # TODO 5a: load the raw training set (no transforms):
    #     raw = datasets.CIFAR10(data_dir, train=True, download=True)
    #   raw.data is a numpy array (50000, 32, 32, 3) uint8; len(raw) == 50000.
    raw = datasets.CIFAR10(data_dir, train=True, download=True)

    # TODO 5b: get (train_idx, val_idx) from split_indices(...).
    train_idx, val_idx = split_indices(len(raw), val_split=val_split, subset_fraction=subset_fraction, seed=seed, leak=leak)

    # TODO 5c: if normalize, compute (mean, std) with compute_mean_std on the TRAINING
    #   images only: raw.data[train_idx]. Otherwise mean, std = None, None.
    #   Question for you: why not compute them on all 50000 images, or on the test set?
    if normalize:
        mean, std = compute_mean_std(raw.data[train_idx])
    else:
        mean, std = None, None

    # TODO 5d: train_tf, eval_tf = build_transforms(...)
    train_tf, eval_tf = build_transforms(normalize=normalize, augment=augment, mean=mean, std=std)

    # TODO 5e: create three CIFAR10 datasets:
    #   train_full: train=True,  transform=train_tf, target_transform=OneHot(smoothing=label_smoothing)
    #   val_full:   train=True,  transform=eval_tf,  target_transform=OneHot()
    #   test_set:   train=False, transform=eval_tf,  target_transform=OneHot(), download=True
    #   (train_full and val_full read the SAME files, but apply different transforms.)
    train_full = datasets.CIFAR10(
        data_dir,
        train=True,
        transform=train_tf,
        target_transform=OneHot(smoothing=label_smoothing),
        download=True,
    )
    val_full = datasets.CIFAR10(
        data_dir,
        train=True,
        transform=eval_tf,
        target_transform=OneHot(),
        download=True,
    )
    test_set = datasets.CIFAR10(
        data_dir,
        train=False,
        transform=eval_tf,
        target_transform=OneHot(),
        download=True,
    )

    # TODO 5f: return Subset(train_full, train_idx), Subset(val_full, val_idx), test_set
    return Subset(train_full, train_idx), Subset(val_full, val_idx), test_set


def make_loaders(train_set, val_set, test_set, batch_size=128, num_workers=2, pin_memory=False):
    """Return (train_loader, val_loader, test_loader). val_loader is None if val_set is empty."""
    # TODO 6: create three DataLoaders with batch_size, num_workers, pin_memory and
    #   persistent_workers=(num_workers > 0).
    #   - ONLY the training loader uses shuffle=True. Why?
    #   - Return None instead of a val loader when len(val_set) == 0.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = None if len(val_set) == 0 else DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


def labels_of(dataset):
    """Integer labels of a CIFAR10 dataset or a Subset of one, without loading any images."""
    if isinstance(dataset, Subset):
        return np.asarray(dataset.dataset.targets)[dataset.indices]
    return np.asarray(dataset.targets)


def class_counts(dataset):
    """Number of images per class, e.g. [900, 912, ...] (length NUM_CLASSES)."""
    # TODO 7a: one line with np.bincount(labels_of(dataset), minlength=...)
    return np.bincount(labels_of(dataset), minlength=NUM_CLASSES)


def count_overlap(train_set, val_set):
    """Number of images that appear in both subsets (should be 0!)."""
    # TODO 7b: both are Subsets -> use their .indices; convert to Python sets and
    #   return the size of the intersection.
    train_idx = set(train_set.indices)
    val_idx = set(val_set.indices)
    return len(train_idx & val_idx)


def channel_stats(dataset, n=1000):
    """Per-channel mean/std of the first n images *after* the dataset's transform.

    With normalize=True the result should be close to mean 0 and std 1. Why?
    """
    # TODO 8: stack the first min(n, len(dataset)) images into one tensor (n, 3, 32, 32)
    #   (dataset[i] returns (image, target)), then reduce over dims (0, 2, 3).
    n = min(n, len(dataset))
    images = []
    for i in range(n):
        image, _ = dataset[i]
        images.append(image)
    images = torch.stack(images)
    mean = images.mean(dim=(0, 2, 3))
    std = images.std(dim=(0, 2, 3))
    return mean, std


def benchmark_loader(loader, n_batches=20):
    """Images per second the loader can deliver (first batch excluded as warm-up)."""
    # TODO 9: measure loader speed.
    #   a) iterator = iter(loader); call next(iterator) once as warm-up (workers start up)
    #   b) start a timer (time.time()), then fetch up to n_batches batches with next(iterator),
    #      counting the images (images.size(0)). Stop early on StopIteration.
    #   c) return images / elapsed seconds   (protect against dividing by 0)
    iterator = iter(loader)
    next(iterator)

    start = time.time()
    total_images = 0
    batches_seen = 0

    for _ in range(n_batches):
        try:
            images, _ = next(iterator)
        except StopIteration:
            break
        total_images += images.size(0)
        batches_seen += 1

    elapsed = time.time() - start
    if elapsed <= 0:
        return 0.0
    return total_images / elapsed


def describe(train_set, val_set, test_set):
    """Print a summary of the three datasets. Expected output: see README "Expected output"."""
    # TODO 10: print, one per line:
    #   a) the sizes of train / val / test
    #   b) the train/val overlap (count_overlap)
    #   c) the class counts of train and (if not empty) val (class_counts(...).tolist())
    #   d) for the first training sample  image, target = train_set[0]:
    #        image shape, dtype and min/max value;  the target vector (target.numpy().round(3))
    #        and its class name CLASSES[target.argmax()]
    #   e) channel_stats of the val set (or the test set if val is empty)
    print(f"train_set size: {len(train_set)}")
    print(f"val_set size: {len(val_set)}")
    print(f"test_set size: {len(test_set)}")
    print(f"train/val overlap: {count_overlap(train_set, val_set)}")
    print(f"train class counts: {class_counts(train_set).tolist()}")
    if len(val_set) > 0:
        print(f"val class counts: {class_counts(val_set).tolist()}")

    image, target = train_set[0]
    print(f"train sample shape: {tuple(image.shape)} dtype={image.dtype} min={image.min().item():.3f} max={image.max().item():.3f}")
    print(f"target: {target.numpy().round(3).tolist()} class={CLASSES[target.argmax()]}")

    if len(val_set) > 0:
        mean, std = channel_stats(val_set)
    else:
        mean, std = channel_stats(test_set)
    print(f"val channel stats: mean={mean.numpy().round(4).tolist()} std={std.numpy().round(4).tolist()}")


def save_augmentation_grid(data_dir, augment, save_path, n_images=6, n_views=6, seed=0):
    """Each row: one original training image followed by n_views random augmentations of it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # TODO 11: draw a grid of n_images rows x (n_views + 1) columns and save it to save_path.
    #   a) train_tf, _ = build_transforms(normalize=False, augment=augment)
    #      (no normalization: we want to SEE natural colors)
    #   b) raw = datasets.CIFAR10(data_dir, train=True, download=True); torch.manual_seed(seed)
    #   c) fig, axes = plt.subplots(n_images, n_views + 1, figsize=(1.6 * (n_views + 1), 1.6 * n_images))
    #   d) for each row: image, label = raw[row]
    #        views = [transforms.ToTensor()(image)] + [train_tf(image) for _ in range(n_views)]
    #        for each (col, view): axes[row, col].imshow(view.permute(1, 2, 0).clamp(0, 1).numpy())
    #                              (imshow wants (H, W, C), tensors are (C, H, W)); hide the axis
    #        put CLASSES[label] as the title of the first image of the row
    #   e) fig.suptitle(...), fig.tight_layout(), fig.savefig(save_path, dpi=120), plt.close(fig)
    train_tf, _ = build_transforms(normalize=False, augment=augment)
    raw = datasets.CIFAR10(data_dir, train=True, download=True)
    torch.manual_seed(seed)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_images, n_views + 1, figsize=(1.6 * (n_views + 1), 1.6 * n_images))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_images):
        image, label = raw[row]
        views = [transforms.ToTensor()(image)] + [train_tf(image) for _ in range(n_views)]
        for col, view in enumerate(views):
            ax = axes[row, col]
            ax.imshow(view.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")
        axes[row, 0].set_title(CLASSES[label])

    fig.suptitle(f"Augmentation grid ({augment})", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
