#!/usr/bin/env python3
"""Debug IQ → STFT pipeline used by run_iq_benchmark.py.

Loads IQMTLDataset (same args as the benchmark), randomly picks ``num_samples``
files from the chosen split (without replacement), then for each prints:

  • Raw IQ tensor shapes before STFT (per task)
  • 2-channel STFT tensor shape after STFT, before bilinear resize
  • Final image shape after resize (should match DataLoader batch)

Saves PNG visualizations (log-magnitude and phase heatmaps) for diagnosis.
Use ``--seed`` for reproducible random file choice and RF train slices; ``--seed -1``
for a fresh random draw each run.

Example::

    python baselines/NeuraCrypt/scripts/debug.py \\
        --pkl_dataset_path /path/to/rf_partition_dict_0.5.pkl \\
        --out_dir baselines/NeuraCrypt/debug_stft_pngs
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

NEURACRYPT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NEURACRYPT_ROOT))

from argparse import Namespace

from sandstone.datasets.iq_dataset import (  # noqa: E402
    IQMTLDataset,
    RF_SLICE_LEN,
    IMG_SIZE,
    _iq_to_image,
    _read_file,
)

# Match iq_dataset._iq_to_image STFT path exactly (for shape + PNG export)


def _stft_pre_resize(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Same STFT + log-mag / phase stack as _iq_to_image, before interpolate."""
    I, Q = x[0], x[1]
    z = torch.complex(I.to(torch.float32), Q.to(torch.float32))
    window = torch.hann_window(n_fft, device=x.device, dtype=z.real.dtype)
    # Must STFT the complex IQ (not z.real) — matches iq_dataset._iq_to_image.
    S = torch.stft(
        z,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        pad_mode="reflect",
        center=True,
    )
    mag = S.abs().clamp(min=1e-9).log()
    phase = S.angle()
    return torch.stack([mag, phase], dim=0).float()


def _slice_rf_for_split(
    rf_x: torch.Tensor, split_group: str, seed: int | None
) -> torch.Tensor:
    """Mirror IQMTLDataset._load RF slicing."""
    if split_group == "train":
        max_start = rf_x.shape[1] - RF_SLICE_LEN
        if seed is not None:
            rng = random.Random(seed)
            start = rng.randint(0, max(0, max_start))
        else:
            start = random.randint(0, max(0, max_start))
        rf_x = rf_x[:, start : start + RF_SLICE_LEN]
    else:
        rf_x = rf_x[:, :RF_SLICE_LEN]
    if rf_x.shape[1] < RF_SLICE_LEN:
        rf_x = F.pad(rf_x, (0, RF_SLICE_LEN - rf_x.shape[1]))
    return rf_x


def _save_stft_png(
    pre_resize: torch.Tensor,
    out_path: str,
    title: str,
) -> None:
    """Save log-mag and phase as a two-panel PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "debug.py needs matplotlib to write PNGs. "
            "Install with: pip install matplotlib"
        ) from e

    mag = pre_resize[0].detach().cpu().numpy()
    ph = pre_resize[1].detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(mag, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("log |STFT|")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(ph, aspect="auto", origin="lower", cmap="twilight")
    axes[1].set_title("phase (rad)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_resized_png(img_256: torch.Tensor, out_path: str, title: str) -> None:
    """Save final (2,256,256) inputs as two-panel PNG (after bilinear resize)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "debug.py needs matplotlib to write PNGs. "
            "Install with: pip install matplotlib"
        ) from e

    mag = img_256[0].detach().cpu().numpy()
    ph = img_256[1].detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(mag, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("log |STFT| (resized)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(ph, aspect="auto", origin="lower", cmap="twilight")
    axes[1].set_title("phase (resized)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Debug IQ STFT shapes + save PNGs")
    default_pkl = (
        "/scratch/10608/aadharsh_aadhithya/data/rep_lr/"
        "OracleDatasetProcessed-arranged/rf_partition_dict_0.5.pkl"
    )
    p.add_argument("--pkl_dataset_path", type=str, default=default_pkl)
    p.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="train",
        help="Which split to read (matches IQMTLDataset)",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Randomly sample this many files from the split (without replacement)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="baselines/NeuraCrypt/debug_stft_pngs",
        help="Directory for PNG outputs",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed: which files are picked + RF train window (train split only). "
        "Use -1 for non-reproducible file choice and RF slice.",
    )
    p.add_argument("--batch_size", type=int, default=2, help="DataLoader batch size")
    return p.parse_args()


def _sample_files(file_list: list, num_samples: int, seed: int | None) -> list:
    """Return up to ``num_samples`` paths chosen uniformly without replacement."""
    n = min(num_samples, len(file_list))
    if n <= 0:
        return []
    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(seed)
    return rng.sample(file_list, k=n)


def main():
    args = parse_args()
    split_group = args.split
    master_seed = None if args.seed < 0 else args.seed

    ds_args = Namespace(
        pkl_dataset_path=args.pkl_dataset_path,
        num_chan=2,
        img_size=[256, 256],
        private_kernel_size=16,
        private_depth=4,
        hidden_dim=2048,
        remove_pixel_shuffle=False,
        load_data_from_encoded_dir=False,
        exam_to_year_dict={},
    )
    IQMTLDataset.set_args(ds_args)
    ds = IQMTLDataset(ds_args, None, split_group)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    pkl = args.pkl_dataset_path
    print("=" * 60)
    print("NeuraCrypt IQ STFT debug")
    print("=" * 60)
    print(f"pkl_dataset_path : {pkl}")
    print(f"split            : {split_group}")
    print(f"num_samples      : {args.num_samples}  (random subset of split)")
    print(f"seed             : {args.seed}  (-1 = non-reproducible)")
    print(f"RF_SLICE_LEN     : {RF_SLICE_LEN}")
    print(f"IMG_SIZE         : {IMG_SIZE}")
    print()

    with open(pkl, "rb") as f:
        meta = pickle.load(f)
    mean_cfo = meta["mean_cfo"]
    std_cfo = meta["std_cfo"]

    key_map = {"train": "train", "dev": "val", "test": "test"}
    file_list = meta[key_map[split_group]]
    picked = _sample_files(file_list, args.num_samples, master_seed)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Split size: {len(file_list)}  →  picked {len(picked)} random path(s).")
    print()

    for i, fp in enumerate(picked):
        print("-" * 60)
        print(f"sample {i} (random draw): {fp}")

        RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y = _read_file(
            fp, mean_cfo, std_cfo
        )

        # Reproducible per-draw RF window on train: different offset per sample index
        rf_slice_seed = (
            (master_seed + 1000 + i) if master_seed is not None else None
        )
        rf_raw = _slice_rf_for_split(RF_X.clone(), split_group, rf_slice_seed)
        tasks = [
            (
                "rf",
                rf_raw,
                dict(n_fft=64, hop_length=8),
            ),
            ("cfo", CFO_X.clone(), dict(n_fft=32, hop_length=4)),
            ("channel", Channel_X.clone(), dict(n_fft=32, hop_length=4)),
        ]

        for name, x_iq, stft_kw in tasks:
            pre = _stft_pre_resize(x_iq, **stft_kw)
            post = _iq_to_image(x_iq, target_size=IMG_SIZE, **stft_kw)
            print(f"  [{name}] raw IQ shape (before STFT)     : {tuple(x_iq.shape)}")
            print(
                f"  [{name}] STFT stack (before resize)      : {tuple(pre.shape)}  "
                f"[channels, F, T]"
            )
            print(
                f"  [{name}] after bilinear → {IMG_SIZE}×{IMG_SIZE} : {tuple(post.shape)}"
            )

            base = os.path.join(out_dir, f"sample{i:03d}_{name}")
            _save_stft_png(
                pre,
                base + "_stft_pre_resize.png",
                f"{name}: STFT (pre-resize) | {os.path.basename(fp)}",
            )
            _save_resized_png(
                post,
                base + "_network_input_resized.png",
                f"{name}: network input (2×{IMG_SIZE}×{IMG_SIZE}) | {os.path.basename(fp)}",
            )
            print(f"  [{name}] saved: {base}_stft_pre_resize.png, {base}_network_input_resized.png")
        print()

    # One DataLoader batch — shapes entering the network (already resized)
    print("-" * 60)
    print("DataLoader batch (tensor shapes as in run_iq_benchmark.py)")
    batch = next(iter(loader))
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:12s} : {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:12s} : {type(v).__name__}")
    print()
    print(f"PNG directory: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
