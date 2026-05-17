"""
Training helpers: data collation, samplers, model I/O, metric logging,
and the OverfitDataLoader for debugging.
"""

import os
import csv
import json
import collections
import time
import datetime
import random
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from torchvision.models.detection.anchor_utils import AnchorGenerator

# ---------------------------------------------------------------------------
# Data collation
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Custom collate that silently drops None items.

    None items arise when a dataset __getitem__ raises a non-fatal error
    (e.g. missing image file). The DataLoader will still produce valid batches
    as long as at least one item per batch is valid.
    """
    batch = [x for x in batch if x is not None]
    if not batch:
        return [], []
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Anchor generator: INPUT FROM CONFIG YAML
# Input: either a list of lists or a tuple of tuples
# ---------------------------------------------------------------------------

def _nested_lists_to_tuples_of_tuples(nested):
    return tuple(tuple(level) for level in nested)
 
def create_anchor_generator(anchor_sizes, aspect_ratios):
    return AnchorGenerator(
        sizes=_nested_lists_to_tuples_of_tuples(anchor_sizes),
        aspect_ratios=_nested_lists_to_tuples_of_tuples(aspect_ratios),
    )

# ---------------------------------------------------------------------------
# Rater weights
# ---------------------------------------------------------------------------

def load_rater_weights(csv_path: str) -> dict:
    """Load and normalise K-alpha weights keyed by annotator ID (e.g. 'R1')."""
    import pandas as pd
    df = pd.read_csv(csv_path).set_index("Annotator")
    raw = df["Avg K-alpha"]
    return (raw / raw.max()).to_dict()


# ---------------------------------------------------------------------------
# Balanced sampler (LVIS-style Repeat Factor Sampling)
# ---------------------------------------------------------------------------
def summarize_balanced_sampler(
    dataset,
    img_ids,
    img_present,
    cat_img_count,
    r_c,
    weights,
    rfs_threshold,
    presence_prob_thresh,
    downweight_nofinding,
    nofinding_weight,
):
    """
    Print a debug summary for Repeat Factor Sampling.

    This is split out from create_balanced_sampler() so the sampling math stays
    readable and the same summary can be reused elsewhere if needed.
    """
    import numpy as np

    N = len(img_ids)
    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}

    print("Creating balanced sampler (RFS)...")
    print(f"  N images: {N}")
    print(f"  dataset: {dataset.__class__.__name__}")
    print(f"  label_mode: {getattr(dataset, 'label_mode', 'unknown')}")
    print(f"  presence_prob_thresh: {presence_prob_thresh}")
    print(f"  rfs_threshold: {rfs_threshold}")
    print(f"  downweight_nofinding: {downweight_nofinding}")
    print(f"  nofinding_weight: {nofinding_weight}")

    num_pure_nofinding = sum(1 for present in img_present if not present)
    print(
        f"  Pure NoFinding images (no disease present): "
        f"{num_pure_nofinding} ({num_pure_nofinding / max(N, 1) * 100:.1f}%)"
    )

    total_present = sum(cat_img_count.values())
    print("\nClass presence in training set (image-level):")
    for class_idx in sorted(cat_img_count.keys()):
        count = cat_img_count[class_idx]
        class_name = idx_to_class.get(class_idx, f"Class {class_idx}")
        print(f"  {class_name} ({class_idx}): {count} images ({count / max(N, 1) * 100:.2f}%)")
    if total_present == 0:
        print("  (no disease classes present under current threshold)")

    print("\nRFS repeat factors (category-level):")
    for class_idx in sorted(r_c.keys()):
        count = cat_img_count[class_idx]
        f_c = count / max(1, N)
        class_name = idx_to_class.get(class_idx, f"Class {class_idx}")
        print(f"  {class_name} ({class_idx}): f_c={f_c:.6f}, r_c={r_c[class_idx]:.3f}")
    if not r_c:
        print("  (no foreground categories received repeat factors)")

    weights_np = np.array(weights, dtype=np.float32)
    nonzero = int((weights_np > 0).sum())

    print("\nImage sampling weights summary:")
    print(f"  nonzero-weight images: {nonzero}/{N}")
    print(f"  min/mean/max: {weights_np.min():.3f} / {weights_np.mean():.3f} / {weights_np.max():.3f}")

    topk = min(10, N)
    if topk > 0:
        top_idx = weights_np.argsort()[-topk:][::-1]
        print(f"\nTop-{topk} sampled images (weight, num_classes_present):")
        for j in top_idx:
            print(f"  {img_ids[j]}: w={weights_np[j]:.3f}, |C|={len(img_present[j])}")

def create_balanced_sampler(
    dataset,
    rfs_threshold=0.01,
    oversample_factor=None,
    presence_prob_thresh=0.1,
    downweight_nofinding=False,
    alpha=0.5,
    nofinding_weight=0.2,
    indices=None,  # NEW: optional list of indices to limit sampling
):
    """
    Repeat Factor Sampling (RFS) for long-tailed detection datasets.

    Args:
        dataset:              Any DetectionDatasetPort-compatible dataset.
        rfs_threshold:        Frequency threshold t (typical: 0.001–0.01).
        presence_prob_thresh: Soft-label disease presence threshold.
        downweight_nofinding: Give pure No-Finding images a low weight.
        alpha:                Exponent for repeat factor (0.5 = sqrt).
        nofinding_weight:     Weight assigned to No-Finding images when
                              ignore_nofinding=True.

    Returns:
        WeightedRandomSampler
    """
    if oversample_factor is not None:
        rfs_threshold = oversample_factor

    img_ids = dataset.get_image_ids()
    # If indices provided, limit to those
    if indices is not None:
        img_ids = [img_ids[i] for i in indices]
    N = len(img_ids)

    img_present = []
    cat_img_count = Counter()

    for img_id in img_ids:
        present = dataset.get_presence_labels(img_id, threshold=presence_prob_thresh)
        img_present.append(present)
        for c in present:
            cat_img_count[c] += 1

    r_c = {}
    for c, cnt in cat_img_count.items():
        f_c = cnt / max(1, N)
        ratio = rfs_threshold / max(f_c, 1e-12)
        r_c[c] = max(1.0, float(ratio ** alpha))

    weights = []
    for present in img_present:
        if not present:
            w = nofinding_weight if downweight_nofinding else 1.0
        else:
            w = max(r_c[c] for c in present)
        weights.append(w)

    summarize_balanced_sampler(
        dataset=dataset,
        img_ids=img_ids,
        img_present=img_present,
        cat_img_count=cat_img_count,
        r_c=r_c,
        weights=weights,
        rfs_threshold=rfs_threshold,
        presence_prob_thresh=presence_prob_thresh,
        downweight_nofinding=downweight_nofinding,
        nofinding_weight=nofinding_weight,
    )
    return WeightedRandomSampler(
        weights=weights, 
        num_samples=int(len(weights)*1.0), # Increase the number of samples to 1.5x the number of images (rare classes see 50% more samples)
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def save_model(model, path: str):
    """Save model state dict to <path>.pth (device-independent)."""
    state = model.to(torch.device("cpu")).state_dict()
    torch.save(state, path + ".pth")


def load_pretrained_weights(network, weights_path: str, device):
    """Load state dict from file into network."""
    state_dict = torch.load(weights_path, map_location=device)
    network.load_state_dict(state_dict)
    return network


def freeze_layers(network, frozen_layers: list):
    """Freeze named parameter groups by substring match."""
    for name, param in network.named_parameters():
        if any(layer_name in name for layer_name in frozen_layers):
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Numpy conversion helper
# ---------------------------------------------------------------------------

def convert_np(obj):
    """Recursively convert numpy scalars/arrays to Python native types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_np(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------

class SmoothedValue:
    """Track a series of values and provide smoothed access."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{name}: {str(meter)}" for name, meter in self.meters.items())

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}",
                 "time: {time}", "data: {data}", "max mem: {memory:.0f}"]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}",
                 "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                    ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


# ---------------------------------------------------------------------------
# OverfitDataLoader — caches N batches for debugging
# ---------------------------------------------------------------------------

class OverfitDataLoader:
    """
    Caches N batches from a DataLoader for overfitting / sanity-check runs.

    Usage::

        loader = OverfitDataLoader(normal_loader, num_batches=5)
        for epoch in range(100):
            for batch in loader:   # same 5 batches every epoch
                train_step(batch)
    """

    def __init__(self, data_loader, num_batches, move_to_device=None):
        self.cached_batches = []
        self.device = move_to_device

        total_batches = len(data_loader)
        if isinstance(num_batches, float) and 0 < num_batches < 1:
            num_batches = max(1, int(total_batches * num_batches))
        num_batches = int(min(num_batches, total_batches))

        print(f"OVERFIT MODE: Caching {num_batches}/{total_batches} batches...")

        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            if self.device is not None:
                images, targets = batch
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                batch = (images, targets)
            self.cached_batches.append(batch)

        self._report_cached_distribution()

    def _report_cached_distribution(self):
        all_labels = []
        for images, targets in self.cached_batches:
            for t in targets:
                all_labels.extend(t["labels"].cpu().tolist())
        dist = Counter(all_labels)
        print(f"Cached batch class distribution: {dict(sorted(dist.items()))}")
        print(f"Total samples in cache: {len(all_labels)}")

    def __iter__(self):
        return iter(self.cached_batches)

    def __len__(self):
        return len(self.cached_batches)
