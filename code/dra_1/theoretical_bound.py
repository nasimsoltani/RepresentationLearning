#!/usr/bin/env python3
"""
Estimate theorem-based reconstruction lower bounds for anisotropic latent noise.

This script matches the repository's current non-isotropic noise path:
- noise is injected on unit-normalized latents z / ||z||
- the aggregate empirical FIM is trace-normalized before eigendecomposition
- beta is interpreted as Tr(Sigma_N), matching noise_level in inject_noise.py

If --target is omitted, bounds are produced for rf, cfo, and channel.
"""

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dra_1.eigenspectrum import get_empirical_fim_per_sample
from dra_1.model_loader import load_model_for_utility_check, load_training_args
from dra_1.py_datasets import ActivationDataset


TARGETS = ("rf", "cfo", "channel")
TARGET_TO_DATASET_KEY = {
    "rf": "RF_X",
    "cfo": "CFO_X",
    "channel": "Channel_X",
}
TASK_NAME_TO_ACTIVATION_DIR = {
    "rf_fingerprinting": "rf",
    "cfo_estimation": "cfo",
    "channel_estimation": "channel",
}
class RunningStats:
    """Streaming first/second moment tracker for flattened tensors."""

    def __init__(self, dim, keep_full_cov):
        self.dim = dim
        self.keep_full_cov = keep_full_cov
        self.count = 0
        self.sum_x = torch.zeros(dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(dim, dtype=torch.float64)
        self.sum_xx = torch.zeros((dim, dim), dtype=torch.float64) if keep_full_cov else None
        self.per_coord_power_sum = 0.0

    def update(self, x_flat):
        x_flat = x_flat.to(torch.float64)
        self.count += x_flat.shape[0]
        self.sum_x += x_flat.sum(dim=0)
        self.sum_sq += (x_flat ** 2).sum(dim=0)
        self.per_coord_power_sum += (x_flat ** 2).mean(dim=1).sum().item()
        if self.keep_full_cov:
            self.sum_xx += x_flat.T @ x_flat

    def finalize(self):
        if self.count == 0:
            raise ValueError("No samples were collected for statistics.")

        mean = self.sum_x / self.count
        second = self.sum_sq / self.count
        var = torch.clamp(second - mean ** 2, min=0.0)
        per_coord_power = self.per_coord_power_sum / self.count

        result = {
            "count": self.count,
            "dim": self.dim,
            "mean": mean,
            "var": var,
            "per_coord_power": per_coord_power,
        }

        if self.keep_full_cov:
            cov = self.sum_xx / self.count - torch.outer(mean, mean)
            cov = 0.5 * (cov + cov.T)
            result["cov"] = cov

        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute theorem-based reconstruction lower bounds for anisotropic noise."
    )
    parser.add_argument("--experiment_path", type=str, required=True, help="Experiment directory with args.json and checkpoint.")
    parser.add_argument(
        "--activation_dir",
        "--activations_path",
        dest="activation_dir",
        type=str,
        default=None,
        help="Activation directory containing the .pth files, for example .../rf_cfo_channel.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to <experiment_path>/theoretical_bounds.")
    parser.add_argument("--target", type=str, choices=TARGETS, default=None, help="Target to analyze. If omitted, loop over rf, cfo, and channel.")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0], help="Noise budgets beta = Tr(Sigma_N).")
    parser.add_argument("--gamma", type=float, default=1e-5, help="Regularizer gamma in sigma_i^2 = c / (lambda_i + gamma).")
    parser.add_argument("--fim_samples", type=int, default=512, help="Number of train samples used to estimate the aggregate FIM.")
    parser.add_argument("--stats_samples", type=int, default=None, help="Number of test samples used for latent/input statistics. Default uses all test files.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for stats collection.")
    parser.add_argument("--fim_batch_size", type=int, default=32, help="Batch size for FIM collection.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subset selection.")
    parser.add_argument("--full_cov_max_dim", type=int, default=1024, help="Use full input covariance only when flattened target dim <= this threshold.")
    parser.add_argument("--stabilizer", type=float, default=1e-8, help="Small stabilizer for variances and log-determinants.")
    parser.add_argument("--actual_noise_type", type=str, default="nonisotropic", choices=["nonisotropic", "isotropic", "none"], help="Attack result family to scan for actual curves.")
    parser.add_argument("--actual_leaked_fraction", type=float, default=None, help="Optional leaked_fraction filter when scanning attack_results_robust.")
    return parser.parse_args()


def resolve_targets(cli_args):
    if cli_args.target is not None:
        return [cli_args.target]
    return list(TARGETS)


def resolve_paths(cli_args, train_args):
    def map_activation_dir_name(tasks):
        if isinstance(tasks, str):
            tasks = [tasks]
        mapped = [TASK_NAME_TO_ACTIVATION_DIR.get(task) for task in tasks]
        mapped = [item for item in mapped if item is not None]
        if not mapped:
            return None
        if len(mapped) == 1:
            return mapped[0]
        return "_".join(mapped)

    def find_run_component(path):
        parts = os.path.normpath(path).split(os.sep)
        for part in parts:
            if re.fullmatch(r"run_\d+", part):
                return part
        return None

    activation_dir_name = map_activation_dir_name(getattr(train_args, "task", None))
    run_name = find_run_component(cli_args.experiment_path)
    activation_base = os.environ.get("ACTIVATIONS_BASE")

    candidates = []
    if cli_args.activation_dir:
        candidates.append(cli_args.activation_dir)
    if activation_base and activation_dir_name:
        if run_name:
            candidates.append(os.path.join(activation_base, run_name, activation_dir_name))
        candidates.append(os.path.join(activation_base, activation_dir_name))
    if run_name and activation_dir_name:
        candidates.append(os.path.join("/scratch/10608/aadharsh_aadhithya/data/rep_lr/activations_exps", run_name, activation_dir_name))
    if activation_dir_name:
        candidates.append(os.path.join(cli_args.experiment_path, "activations"))
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(cli_args.experiment_path))), "activations", activation_dir_name))

    activation_dir = None
    tried = []
    for candidate in candidates:
        if candidate and candidate not in tried:
            tried.append(candidate)
            if os.path.isdir(candidate):
                activation_dir = candidate
                break

    if activation_dir is None:
        raise FileNotFoundError(
            "Activations directory not found. Tried: " + ", ".join(tried)
        )

    output_dir = cli_args.output_dir or os.path.join(cli_args.experiment_path, "theoretical_bounds")
    os.makedirs(output_dir, exist_ok=True)

    partition_path = getattr(train_args, "pkl_dataset_path", None)
    if partition_path is None or not os.path.exists(partition_path):
        fallback = os.path.join(os.getcwd(), "dataset", "rf_partition_dict_0.5.pkl")
        if os.path.exists(fallback):
            partition_path = fallback
        else:
            raise FileNotFoundError("Could not resolve partition pickle path.")

    return activation_dir, output_dir, partition_path


def load_partitions(partition_path):
    with open(partition_path, "rb") as handle:
        partitions = pickle.load(handle)
    for split in ("train", "test"):
        if split not in partitions:
            raise KeyError(f"Partition file missing '{split}' split: {partition_path}")
    return partitions


def choose_subset(file_list, max_count, rng):
    file_list = list(file_list)
    if max_count is None or max_count >= len(file_list):
        return file_list
    indices = list(range(len(file_list)))
    rng.shuffle(indices)
    chosen = sorted(indices[:max_count])
    return [file_list[i] for i in chosen]


def flatten_target_tensor(target_name, batch):
    rf_x, _, cfo_x, _, channel_x, _, _, _ = batch
    if target_name == "rf":
        tensor = rf_x
    elif target_name == "cfo":
        tensor = cfo_x
    else:
        tensor = channel_x

    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)
    return tensor.reshape(tensor.shape[0], -1)


def flatten_activation_tensor(activations):
    if activations.dim() == 4 and activations.size(1) == 1:
        activations = activations.squeeze(1)
    return activations.reshape(activations.shape[0], -1)


def compute_aggregate_fim(heads, activation_dir, train_files, latent_dim, cli_args):
    rng = random.Random(cli_args.seed)
    fim_files = choose_subset(train_files, cli_args.fim_samples, rng)
    dataset = ActivationDataset(activation_dir=activation_dir, file_list=fim_files, test_mode=True)
    loader = DataLoader(dataset, batch_size=cli_args.fim_batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fim_total = torch.zeros((latent_dim, latent_dim), device=device, dtype=torch.float64)
    sample_count = 0

    for batch in tqdm(loader, desc="Estimating aggregate FIM", ncols=80):
        _, rf_labels, _, cfo_labels, _, ch_labels, activations, _ = batch
        batch_size = activations.size(0)

        for idx in range(batch_size):
            sample_activation = activations[idx:idx + 1]
            fim_sample = torch.zeros((latent_dim, latent_dim), device=device, dtype=torch.float64)
            valid = False

            for task_name, head in heads.items():
                if task_name == "rf_fingerprinting":
                    labels = rf_labels[idx:idx + 1]
                elif task_name == "cfo_estimation":
                    labels = cfo_labels[idx:idx + 1]
                elif task_name == "channel_estimation":
                    labels = ch_labels[idx:idx + 1]
                else:
                    continue

                fim_task = get_empirical_fim_per_sample(
                    head=head,
                    activations=sample_activation,
                    labels=labels,
                    device=device,
                    latent_dim=latent_dim,
                )
                if fim_task is not None:
                    fim_sample += fim_task.to(torch.float64)
                    valid = True

            if valid:
                fim_total += fim_sample
                sample_count += 1

    if sample_count == 0:
        raise ValueError("No valid samples were available for FIM estimation.")

    fim_avg = fim_total / sample_count
    trace = torch.trace(fim_avg)
    if trace <= cli_args.stabilizer:
        raise ValueError("Aggregate FIM trace is too small for a stable eigendecomposition.")

    fim_norm = fim_avg / trace
    eigenvalues, eigenvectors = torch.linalg.eigh(fim_norm)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    return {
        "fim": fim_norm.detach().cpu(),
        "trace_before_norm": float(trace.item()),
        "sample_count": sample_count,
        "eigenvalues": eigenvalues.detach().cpu(),
        "eigenvectors": eigenvectors.detach().cpu(),
    }


def collect_latent_and_input_stats(activation_dir, test_files, targets, latent_dim, cli_args):
    rng = random.Random(cli_args.seed + 1)
    stats_files = choose_subset(test_files, cli_args.stats_samples, rng)
    dataset = ActivationDataset(activation_dir=activation_dir, file_list=stats_files, test_mode=True)
    loader = DataLoader(dataset, batch_size=cli_args.batch_size, shuffle=False)

    z_stats = RunningStats(dim=latent_dim, keep_full_cov=True)
    target_stats = {}

    for batch in tqdm(loader, desc="Collecting latent/input stats", ncols=80):
        _, _, _, _, _, _, activations, _ = batch
        z_flat = flatten_activation_tensor(activations).to(torch.float64)
        z_norm = torch.linalg.norm(z_flat, dim=1, keepdim=True)
        z_unit = z_flat / (z_norm + cli_args.stabilizer)
        z_stats.update(z_unit)

        for target in targets:
            x_flat = flatten_target_tensor(target, batch)
            if target not in target_stats:
                keep_full_cov = x_flat.shape[1] <= cli_args.full_cov_max_dim
                target_stats[target] = RunningStats(dim=x_flat.shape[1], keep_full_cov=keep_full_cov)
            target_stats[target].update(x_flat)

    z_result = z_stats.finalize()
    sigma_z = z_result["cov"]
    sigma_z = 0.5 * (sigma_z + sigma_z.T)

    return {
        "sigma_z": sigma_z,
        "z_count": z_result["count"],
        "target_stats": {name: tracker.finalize() for name, tracker in target_stats.items()},
    }


def compute_bound_curve(eigenvalues, eigenvectors, sigma_z, input_stats, betas, gamma, stabilizer):
    latent_dim = eigenvalues.numel()
    identity = torch.eye(latent_dim, dtype=torch.float64)

    sigma_z = sigma_z.to(torch.float64)
    eigenvalues = eigenvalues.to(torch.float64)
    eigenvectors = eigenvectors.to(torch.float64)

    diag_var = torch.clamp(input_stats["var"].to(torch.float64), min=stabilizer)
    per_coord_power = float(input_stats["per_coord_power"])
    d = int(input_stats["dim"])

    diag_logdet_sigma_x = torch.log(diag_var).sum().item()
    dense_logdet_sigma_x = None
    if "cov" in input_stats:
        cov = input_stats["cov"].to(torch.float64)
        cov = 0.5 * (cov + cov.T) + stabilizer * torch.eye(cov.shape[0], dtype=torch.float64)
        sign, logabsdet = torch.linalg.slogdet(cov)
        if sign > 0:
            dense_logdet_sigma_x = float(logabsdet.item())

    sum_inv = torch.sum(1.0 / (eigenvalues + gamma))
    results = []

    for beta in betas:
        if beta < 0:
            raise ValueError("Beta values must be non-negative.")

        if beta == 0:
            sigma2 = torch.zeros_like(eigenvalues)
            logdet_m = float("inf")
            xi_diag = 0.0
            nmse_diag = 0.0
            xi_full = 0.0 if dense_logdet_sigma_x is not None else None
            nmse_full = 0.0 if dense_logdet_sigma_x is not None else None
        else:
            c = beta / sum_inv
            sigma2 = c / (eigenvalues + gamma)
            inv_sqrt_sigma2 = torch.rsqrt(torch.clamp(sigma2, min=stabilizer))
            sigma_inv_half = eigenvectors @ torch.diag(inv_sqrt_sigma2) @ eigenvectors.T
            m_matrix = identity + sigma_inv_half @ sigma_z @ sigma_inv_half
            m_matrix = 0.5 * (m_matrix + m_matrix.T)
            sign_m, logdet_tensor = torch.linalg.slogdet(m_matrix)
            if sign_m <= 0:
                raise ValueError(f"logdet(I + Sigma_N^-1/2 Sigma_z Sigma_N^-1/2) is not positive for beta={beta}")
            logdet_m = float(logdet_tensor.item())

            log_xi_diag = (diag_logdet_sigma_x - logdet_m) / d
            xi_diag = float(math.exp(log_xi_diag))
            nmse_diag = float(xi_diag / max(per_coord_power, stabilizer))

            if dense_logdet_sigma_x is not None:
                log_xi_full = (dense_logdet_sigma_x - logdet_m) / d
                xi_full = float(math.exp(log_xi_full))
                nmse_full = float(xi_full / max(per_coord_power, stabilizer))
            else:
                xi_full = None
                nmse_full = None

        results.append(
            {
                "beta": float(beta),
                "trace_sigma_n": float(beta),
                "sigma2_min": float(sigma2.min().item()) if beta > 0 else 0.0,
                "sigma2_max": float(sigma2.max().item()) if beta > 0 else 0.0,
                "logdet_m": logdet_m,
                "xi_lb_diag": xi_diag,
                "nmse_lb_diag": nmse_diag,
                "xi_lb_full": xi_full,
                "nmse_lb_full": nmse_full,
            }
        )

    return {
        "input_dim": d,
        "per_coord_power": per_coord_power,
        "diag_logdet_sigma_x": diag_logdet_sigma_x,
        "full_logdet_sigma_x": dense_logdet_sigma_x,
        "curve": results,
    }


def parse_mse_values(result_path):
    mse_values = []
    with open(result_path, "r") as handle:
        for line in handle:
            match = re.search(r'"MSE"\s*:\s*([-+0-9.eE]+)', line)
            if match:
                mse_values.append(float(match.group(1)))
    return mse_values


def parse_attack_metadata_from_path(path):
    path = path.replace(os.sep, "/")
    noise_match = re.search(r"/(nonisotropic|isotropic|none)/", path)
    leaked_match = re.search(r"/(?:leaked_frac|frac)_([0-9_]+)/", path)
    beta_match = re.search(r"/(?:noise_level|level)_([0-9_]+)/", path)

    noise_type = noise_match.group(1) if noise_match else None
    leaked_fraction = None
    beta = None

    if leaked_match:
        leaked_fraction = float(leaked_match.group(1).replace("_", "."))
    if beta_match:
        beta = float(beta_match.group(1).replace("_", "."))

    return noise_type, leaked_fraction, beta


def discover_actual_curves(experiment_path, target, per_coord_power, cli_args):
    attack_root = os.path.join(experiment_path, "attack_results_robust", target)
    if not os.path.isdir(attack_root):
        return []

    grouped = defaultdict(list)
    expected_result_name = f"attack_{target}_results.json"

    for root, _, files in os.walk(attack_root):
        if expected_result_name not in files:
            continue

        result_path = os.path.join(root, expected_result_name)
        noise_type, leaked_fraction, beta = parse_attack_metadata_from_path(result_path)

        attack_args = None
        args_path = os.path.join(root, "attack_args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as handle:
                attack_args = json.load(handle)
            noise_type = attack_args.get("noise_type", noise_type)
            leaked_fraction = attack_args.get("leaked_fraction", leaked_fraction)
            beta = attack_args.get("noise_level", beta)

        if noise_type != cli_args.actual_noise_type:
            continue

        if cli_args.actual_leaked_fraction is not None:
            if leaked_fraction is None or abs(float(leaked_fraction) - cli_args.actual_leaked_fraction) > 1e-12:
                continue
        if beta is None:
            continue

        mse_values = parse_mse_values(result_path)
        if not mse_values:
            continue

        beta = float(beta)
        avg_mse = float(sum(mse_values) / len(mse_values))
        avg_nmse = float(avg_mse / max(per_coord_power, cli_args.stabilizer))
        series_key = f"leaked_fraction={leaked_fraction}"
        grouped[series_key].append(
            {
                "beta": beta,
                "avg_mse": avg_mse,
                "avg_nmse": avg_nmse,
                "num_samples": len(mse_values),
                "path": root,
            }
        )

    series = []
    for label, points in grouped.items():
        points.sort(key=lambda item: item["beta"])
        series.append({"label": label, "points": points})

    series.sort(key=lambda item: item["label"])
    return series


def plot_curves(target, bound_data, actual_series, output_path):
    curve = bound_data["curve"]
    betas = [entry["beta"] for entry in curve if entry["beta"] > 0]
    nmse_diag = [entry["nmse_lb_diag"] for entry in curve if entry["beta"] > 0]
    nmse_full = [entry["nmse_lb_full"] for entry in curve if entry["beta"] > 0 and entry["nmse_lb_full"] is not None]
    betas_full = [entry["beta"] for entry in curve if entry["beta"] > 0 and entry["nmse_lb_full"] is not None]

    plt.figure(figsize=(8, 5))
    if betas:
        plt.loglog(betas, nmse_diag, "o-", linewidth=2, label="Bound (diag entropy surrogate)")
    if betas_full:
        plt.loglog(betas_full, nmse_full, "s--", linewidth=2, label="Bound (full Gaussian surrogate)")

    for series in actual_series:
        series_betas = [point["beta"] for point in series["points"] if point["beta"] > 0]
        series_nmse = [point["avg_nmse"] for point in series["points"] if point["beta"] > 0]
        if series_betas:
            plt.loglog(series_betas, series_nmse, "x-.", linewidth=1.5, label=f"Actual ({series['label']})")

    plt.xlabel("beta = Tr(Sigma_N)")
    plt.ylabel("NMSE")
    plt.title(f"{target.upper()} reconstruction: theorem bound vs actual")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_target_outputs(target, output_dir, payload, actual_series):
    target_dir = os.path.join(output_dir, target)
    os.makedirs(target_dir, exist_ok=True)

    json_path = os.path.join(target_dir, "theoretical_bound.json")
    plot_path = os.path.join(target_dir, "theoretical_bound.png")

    payload["actual_curves"] = actual_series
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    plot_curves(target, payload["bound"], actual_series, plot_path)
    return json_path, plot_path


def main():
    cli_args = parse_args()
    targets = resolve_targets(cli_args)

    train_args = load_training_args(cli_args.experiment_path)
    activation_dir, output_dir, partition_path = resolve_paths(cli_args, train_args)
    partitions = load_partitions(partition_path)

    model_data = load_model_for_utility_check(cli_args.experiment_path)
    if not model_data["is_mtl"]:
        raise ValueError("This script currently expects a multi-task checkpoint with all task heads loaded.")

    sample_activation_dir = activation_dir
    sample_files = [f for f in os.listdir(sample_activation_dir) if f.endswith(".pth")]
    if not sample_files:
        raise ValueError(f"No activation files found in {sample_activation_dir}")
    sample = torch.load(os.path.join(sample_activation_dir, sample_files[0]), map_location="cpu")
    latent_dim = int(np.prod(sample["activation"].shape))

    fim_data = compute_aggregate_fim(
        heads=model_data["model"]["heads"],
        activation_dir=activation_dir,
        train_files=partitions["train"],
        latent_dim=latent_dim,
        cli_args=cli_args,
    )

    stats_data = collect_latent_and_input_stats(
        activation_dir=activation_dir,
        test_files=partitions["test"],
        targets=targets,
        latent_dim=latent_dim,
        cli_args=cli_args,
    )

    summary = {
        "experiment_path": cli_args.experiment_path,
        "activation_dir": activation_dir,
        "partition_path": partition_path,
        "targets": targets,
        "betas": [float(beta) for beta in cli_args.betas],
        "gamma": float(cli_args.gamma),
        "latent_dim": latent_dim,
        "fim": {
            "num_samples": fim_data["sample_count"],
            "trace_before_norm": fim_data["trace_before_norm"],
            "eigenvalue_min": float(fim_data["eigenvalues"].min().item()),
            "eigenvalue_max": float(fim_data["eigenvalues"].max().item()),
        },
        "normalized_latent_note": "Sigma_z is computed on z / ||z|| to match inject_noise.py and the current evaluation path.",
        "targets_written": {},
    }

    for target in targets:
        input_stats = stats_data["target_stats"][target]
        bound = compute_bound_curve(
            eigenvalues=fim_data["eigenvalues"],
            eigenvectors=fim_data["eigenvectors"],
            sigma_z=stats_data["sigma_z"],
            input_stats=input_stats,
            betas=cli_args.betas,
            gamma=cli_args.gamma,
            stabilizer=cli_args.stabilizer,
        )
        actual_series = discover_actual_curves(
            experiment_path=cli_args.experiment_path,
            target=target,
            per_coord_power=bound["per_coord_power"],
            cli_args=cli_args,
        )

        payload = {
            "target": target,
            "bound": bound,
            "meta": {
                "input_key": TARGET_TO_DATASET_KEY[target],
                "stats_count": input_stats["count"],
                "used_full_covariance": "cov" in input_stats,
                "full_cov_max_dim": cli_args.full_cov_max_dim,
            },
        }
        json_path, plot_path = write_target_outputs(target, output_dir, payload, actual_series)
        summary["targets_written"][target] = {
            "json": json_path,
            "plot": plot_path,
            "actual_series_found": len(actual_series),
        }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
