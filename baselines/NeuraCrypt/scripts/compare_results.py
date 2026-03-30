#!/usr/bin/env python3
"""Compare NeuraCrypt IQ results against the rep_lr MTL baseline.

Usage::

    python scripts/compare_results.py \
        --neuracrypt_json  results_iq/neuracrypt_results.json \
        --reprlr_json      /path/to/rep_lr_results.json \
        [--output_fig      comparison.pdf]

``reprlr_json`` should be a JSON (or can be omitted to use hard-coded values
from the user's table).  The expected format is::

    {
      "cfo":     {"final_mse": 0.378, "rf_accuracy": 0.460, "cfo_r2": 0.218, "channel_r2": 0.347},
      "channel": {"final_mse": 0.436, ...},
      "rf":      {"final_mse": 0.501, ...}
    }

or simply a flat dict with the four keys.
"""

import argparse
import json
import os
import sys

import numpy as np

# optional matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ──────────────────────────────────────────────────────────────────────────────
# Hard-coded rep_lr results from the paper run (used as fallback)
# ──────────────────────────────────────────────────────────────────────────────
REP_LR_DEFAULTS = {
    'rf_accuracy': 0.46016,
    'cfo_r2':      0.217948,
    'channel_r2':  0.346807,
    'final_mse':   (0.378185 + 0.436039 + 0.500942) / 3,   # average over 3 tasks
}

METRICS = [
    ('rf_accuracy', 'RF Accuracy ↑',   True),
    ('cfo_r2',      'CFO R² ↑',        True),
    ('channel_r2',  'Channel R² ↑',    True),
    ('final_mse',   'Final MSE ↓',     False),
]


def parse_args():
    p = argparse.ArgumentParser(description='Compare NeuraCrypt vs rep_lr')
    p.add_argument('--neuracrypt_json', required=True,
                   help='Path to results_iq/neuracrypt_results.json')
    p.add_argument('--reprlr_json', default=None,
                   help='Path to rep_lr results JSON (optional)')
    p.add_argument('--output_fig', default=None,
                   help='Output figure path (.pdf or .png)')
    return p.parse_args()


def load_neuracrypt(path: str) -> dict:
    data = json.load(open(path))
    return data.get('summary', data)


def load_reprlr(path: str) -> dict:
    data = json.load(open(path))
    # Accept both flat dict and task-keyed dict
    if 'rf_accuracy' in data:
        return data
    # Average across tasks if task-keyed
    flat = {}
    for key in ['rf_accuracy', 'cfo_r2', 'channel_r2', 'final_mse']:
        vals = [v[key] for v in data.values() if key in v]
        flat[key] = float(np.mean(vals)) if vals else float('nan')
    return flat


def print_table(nc: dict, rl: dict):
    header = f"{'Metric':<22} {'rep_lr (MTL)':>16} {'NeuraCrypt':>16} {'Δ':>10}"
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for key, label, higher_better in METRICS:
        rl_val = rl.get(key, float('nan'))
        nc_val = nc.get(key, float('nan'))
        delta  = nc_val - rl_val
        arrow  = '↑' if (delta > 0) == higher_better else '↓'
        print(f'{label:<22} {rl_val:>16.4f} {nc_val:>16.4f} {delta:>+9.4f} {arrow}')
    print(sep + '\n')


def plot_comparison(nc: dict, rl: dict, out_path: str):
    if not HAS_MPL:
        print('[INFO] matplotlib not available; skipping figure.')
        return

    labels = [lbl for _, lbl, _ in METRICS]
    nc_vals = [nc.get(k, float('nan')) for k, _, _ in METRICS]
    rl_vals = [rl.get(k, float('nan')) for k, _, _ in METRICS]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, rl_vals, width, label='rep_lr (MTL)', color='steelblue')
    bars2 = ax.bar(x + width/2, nc_vals, width, label='NeuraCrypt',   color='coral')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('rep_lr (MTL) vs NeuraCrypt on IQ Tasks')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'Figure saved to {out_path}')
    plt.close()


def main():
    args = parse_args()

    nc = load_neuracrypt(args.neuracrypt_json)
    rl = load_reprlr(args.reprlr_json) if args.reprlr_json else REP_LR_DEFAULTS

    print('\n=== rep_lr Results ===')
    for k, v in rl.items():
        print(f'  {k}: {v:.4f}')

    print('\n=== NeuraCrypt Results ===')
    for k, v in nc.items():
        if isinstance(v, (int, float)):
            print(f'  {k}: {v:.4f}')

    print_table(nc, rl)

    if args.output_fig:
        plot_comparison(nc, rl, args.output_fig)


if __name__ == '__main__':
    main()
