"""
Utilities to turn prebuilt numpy windows
(X_train, Y_type_train, Y_actor_train), (X_val, ...), (X_test, ...)
into PyTorch Datasets & DataLoaders with WeightedRandomSampler,
plus handy save/load helpers.

Usage (example):

from config import TrainCfg, ModelCfg
from dataset_build import (
    WindowDataset, make_dataloaders, save_numpy_artifacts, load_numpy_artifacts
)


"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ------------------------------
# Dataset
# ------------------------------
class WindowDataset(Dataset):
    """Lightweight dataset over numpy (or memmap) arrays.

    - Avoids doubling memory by default (keeps numpy and converts per item).
    - Optionally move all to tensors up-front with in_memory=True.
    """
    def __init__(
        self,
        X: np.ndarray,
        Y_type: np.ndarray,
        Y_actor: np.ndarray,
        *,
        dtype: torch.dtype = torch.float32,
        in_memory: bool = False,
    ) -> None:
        assert X.ndim == 4, f"X must be (B,T,N,F), got {X.shape}"
        assert Y_type.shape[0] == X.shape[0] and Y_actor.shape[0] == X.shape[0]
        self.in_memory = in_memory
        self.dtype = dtype

        if in_memory:
            # Convert once to tensors
            self.X = torch.from_numpy(X).to(dtype)
            self.Y_type = torch.from_numpy(Y_type).long()
            self.Y_actor = torch.from_numpy(Y_actor).long()
        else:
            # Keep numpy/memmap to save RAM; convert per __getitem__
            self.X = X
            self.Y_type = Y_type
            self.Y_actor = Y_actor

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.in_memory:
            return self.X[idx], self.Y_type[idx], self.Y_actor[idx]
        # Convert on the fly
        x = torch.from_numpy(self.X[idx]).to(self.dtype)
        yt = torch.as_tensor(self.Y_type[idx], dtype=torch.long)
        ya = torch.as_tensor(self.Y_actor[idx], dtype=torch.long)
        return x, yt, ya


# ------------------------------
# Sampler / class weights
# ------------------------------
@dataclass
class SamplerCfg:
    max_weight_clip: float = 50.0  # to avoid extreme oversampling


def build_class_weights(y_type_train: np.ndarray, n_types: int, device: torch.device, max_clip: float = 50.0) -> torch.Tensor:
    """Balanced class weights: w_c = N / (K * n_c), normalized to mean=1 and clipped."""
    y = torch.from_numpy(y_type_train)
    counts = torch.bincount(y, minlength=n_types).float().to(device)
    N, K = float(len(y)), float(n_types)
    w = N / (K * counts)
    w = (w / w.mean()).clamp(max=max_clip)
    return w


def build_weighted_sampler(y_type_train: np.ndarray, n_types: int, max_clip: float = 50.0) -> WeightedRandomSampler:
    """WeightedRandomSampler over samples based on inverse class frequency."""
    y = torch.from_numpy(y_type_train)
    counts = torch.bincount(y, minlength=n_types).float()
    N, K = float(len(y)), float(n_types)
    class_w = N / (K * counts)
    class_w = (class_w / class_w.mean()).clamp(max=max_clip)
    sample_w = class_w[y]
    return WeightedRandomSampler(weights=sample_w, num_samples=len(y), replacement=True)


# ------------------------------
# Dataloaders
# ------------------------------

def make_dataloaders(
    X_train: np.ndarray, Yt_train: np.ndarray, Ya_train: np.ndarray,
    X_val:   np.ndarray, Yt_val:   np.ndarray, Ya_val:   np.ndarray,
    X_test:  np.ndarray, Yt_test:  np.ndarray, Ya_test:  np.ndarray,
    *,
    batch_size: int = 64,
    n_types: int = 6,
    use_weighted_sampler: bool = True,
    sampler_max_weight_clip: float = 50.0,
    in_memory: bool = False,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[WeightedRandomSampler]]:
    """Create train/val/test DataLoaders. Returns (train, val, test, sampler)."""
    # Build datasets
    train_ds = WindowDataset(X_train, Yt_train, Ya_train, dtype=dtype, in_memory=in_memory)
    val_ds   = WindowDataset(X_val,   Yt_val,   Ya_val,   dtype=dtype, in_memory=in_memory)
    test_ds  = WindowDataset(X_test,  Yt_test,  Ya_test,  dtype=dtype, in_memory=in_memory)

    # Sampler (train only)
    sampler = None
    if use_weighted_sampler:
        tqdm.write("Building WeightedRandomSampler for training...")
        sampler = build_weighted_sampler(Yt_train, n_types, max_clip=sampler_max_weight_clip)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader= DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, sampler


# ------------------------------
# Save / Load helpers
# ------------------------------

def save_numpy_artifacts(
    out_dir: str,
    X_train: np.ndarray, Yt_train: np.ndarray, Ya_train: np.ndarray,
    X_val:   np.ndarray, Yt_val:   np.ndarray, Ya_val:   np.ndarray,
    X_test:  np.ndarray, Yt_test:  np.ndarray, Ya_test:  np.ndarray,
    *,
    use_float16: bool = False,
) -> None:
    """Save arrays as .npy in out_dir. Optionally cast X_* to float16 to save disk/RAM."""
    os.makedirs(out_dir, exist_ok=True)

    def maybe_cast(X):
        return X.astype(np.float16) if use_float16 and X.dtype != np.float16 else X

    np.save(os.path.join(out_dir, 'X_train.npy'), maybe_cast(X_train))
    np.save(os.path.join(out_dir, 'Y_type_train.npy'), Yt_train)
    np.save(os.path.join(out_dir, 'Y_actor_train.npy'), Ya_train)

    np.save(os.path.join(out_dir, 'X_val.npy'),   maybe_cast(X_val))
    np.save(os.path.join(out_dir, 'Y_type_val.npy'),   Yt_val)
    np.save(os.path.join(out_dir, 'Y_actor_val.npy'),  Ya_val)

    np.save(os.path.join(out_dir, 'X_test.npy'),  maybe_cast(X_test))
    np.save(os.path.join(out_dir, 'Y_type_test.npy'),  Yt_test)
    np.save(os.path.join(out_dir, 'Y_actor_test.npy'), Ya_test)

    print(f"Saved numpy artifacts to {out_dir}")


def load_numpy_artifacts(base_dir: str, memmap_mode: Optional[str] = None):
    """Load arrays back. If memmap_mode is provided (e.g., 'r'), returns np.memmap for X_*.
    This is useful to stream large data from disk without loading fully into RAM.
    """
    def load_X(name: str):
        path = os.path.join(base_dir, name)
        if memmap_mode is None:
            return np.load(path)
        arr = np.load(path, mmap_mode=memmap_mode)
        return arr

    X_train = load_X('X_train.npy')
    Yt_train = np.load(os.path.join(base_dir, 'Y_type_train.npy'))
    Ya_train = np.load(os.path.join(base_dir, 'Y_actor_train.npy'))

    X_val   = load_X('X_val.npy')
    Yt_val  = np.load(os.path.join(base_dir, 'Y_type_val.npy'))
    Ya_val  = np.load(os.path.join(base_dir, 'Y_actor_val.npy'))

    X_test  = load_X('X_test.npy')
    Yt_test = np.load(os.path.join(base_dir, 'Y_type_test.npy'))
    Ya_test = np.load(os.path.join(base_dir, 'Y_actor_test.npy'))

    return (X_train, Yt_train, Ya_train), (X_val, Yt_val, Ya_val), (X_test, Yt_test, Ya_test)


# ------------------------------
# Optional: quick sanity summaries
# ------------------------------

def print_memory_footprint(**named_arrays: np.ndarray) -> None:
    """Pretty-print RAM usage of given numpy arrays."""
    def fmt_bytes(n: int) -> str:
        for unit in ["B","KiB","MiB","GiB","TiB"]:
            if n < 1024 or unit == "TiB":
                return f"{n:.2f} {unit}"
            n /= 1024
    for k, v in named_arrays.items():
        print(f"{k:16s} -> {fmt_bytes(v.nbytes)}")
