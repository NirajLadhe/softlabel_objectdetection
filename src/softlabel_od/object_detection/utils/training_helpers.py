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

def create_balanced_sampler(
    dataset,
    rfs_threshold=0.01,
    presence_prob_thresh=0.1,
    ignore_nofinding=False,
    alpha=0.5,
    nofinding_weight=0.1,
):
    """
    Repeat Factor Sampling (RFS) for long-tailed detection datasets.

    Args:
        dataset:              A Vin_Dr_CXR_SoftLabel_Dataset instance.
        rfs_threshold:        Frequency threshold t (typical: 0.001–0.01).
        presence_prob_thresh: Soft-label disease presence threshold.
        ignore_nofinding:     Give pure No-Finding images a low weight.
        alpha:                Exponent for repeat factor (0.5 = sqrt).
        nofinding_weight:     Weight assigned to No-Finding images when
                              ignore_nofinding=True.

    Returns:
        WeightedRandomSampler
    """
    img_ids = dataset.image_ids
    N = len(img_ids)

    img_present = []
    cat_img_count = Counter()

    for img_id in img_ids:
        labels = dataset.image_data[img_id]
        present = set()

        if dataset.dataset_type == "soft":
            for cl in labels:
                mean_box = (cl.get("spatial") or {}).get("mean")
                if mean_box and mean_box[2] >= 0.99 and mean_box[3] >= 0.99:
                    continue
                probs = cl.get("class_probs", {}) or {}
                for name, p in probs.items():
                    if name in dataset.class_to_idx and p >= presence_prob_thresh:
                        present.add(dataset.class_to_idx[name])
        else:
            for finding in labels:
                if "renormalized_class_probs" in finding:
                    for name, p in (finding.get("renormalized_class_probs") or {}).items():
                        if name in dataset.class_to_idx and p >= presence_prob_thresh:
                            present.add(dataset.class_to_idx[name])
                elif "class" in finding and finding["class"] in dataset.class_to_idx:
                    present.add(dataset.class_to_idx[finding["class"]])

        img_present.append(present)
        for c in present:
            cat_img_count[c] += 1

    r_c = {}
    for c, cnt in cat_img_count.items():
        f_c = cnt / max(1, N)
        ratio = rfs_threshold / max(f_c, 1e-12)
        r_c[c] = max(1.0, float(ratio ** (alpha / 2.0)))

    weights = []
    for present in img_present:
        if not present:
            w = nofinding_weight if ignore_nofinding else 1.0
        else:
            w = max(r_c[c] for c in present)
        weights.append(w)

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


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
