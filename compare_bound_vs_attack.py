"""
Compare Theorem 1 empirical lower bounds (theoretical_bounds/*/theoretical_bound.json)
against actual attack MSEs (best decoder val loss in attack_results_robust/**/epoch_history.json)
for the same experiment.
"""
import json
import os
import re
import sys
from collections import defaultdict

EXP = sys.argv[1] if len(sys.argv) > 1 else (
    "/scratch/10608/aadharsh_aadhithya/results/rep_lr/results_20250725_111527/run_1/"
    "rf_cfo_channel/rf_fingerprinting_cfo_estimation_channel_estimation_20250725_234643"
)

TARGETS = ("rf", "cfo", "channel")


def parse_float_token(token):
    return float(token.replace("_", "."))


def collect_attacks(exp_path):
    """Walk attack_results_robust, return {(task, noise_type, leaked, beta, lam): [best_val_losses]}"""
    root = os.path.join(exp_path, "attack_results_robust")
    out = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        if "epoch_history.json" not in files:
            continue
        rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        m = re.search(
            r"(?:^|/)(rf|cfo|channel)/(isotropic|nonisotropic|none)/"
            r"leaked_frac_([0-9_]+)/noise_level_([0-9_]+)(?:/lambda_([0-9_]+))?",
            rel,
        )
        if not m:
            continue
        task, ntype, leaked, beta, lam = m.groups()
        leaked = parse_float_token(leaked)
        beta = parse_float_token(beta)
        lam = parse_float_token(lam) if lam else None
        try:
            with open(os.path.join(dirpath, "epoch_history.json")) as fh:
                hist = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not hist:
            continue
        best = min(e["val_loss"] for e in hist if "val_loss" in e)
        out[(task, ntype, leaked, beta, lam)].append(best)
    return out


def load_bounds(exp_path):
    """Return {task: {beta: bound_dict}} from theoretical_bound.json files."""
    bounds = {}
    for target in TARGETS:
        p = os.path.join(exp_path, "theoretical_bounds", target, "theoretical_bound.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            data = json.load(fh)
        curve = {entry["beta"]: entry for entry in data["bound"]["curve"]}
        bounds[target] = {
            "curve": curve,
            "per_coord_power": data["bound"]["per_coord_power"],
            "input_dim": data["bound"]["input_dim"],
        }
    return bounds


def main():
    print(f"Experiment: {EXP}\n")
    attacks = collect_attacks(EXP)
    bounds = load_bounds(EXP)

    if not attacks:
        print("No attack results found.")
        return
    if not bounds:
        print("No theoretical bound files found.")
        return

    # organize attacks: task -> ntype -> beta -> list of (leaked, lam, mse)
    per_task = defaultdict(lambda: defaultdict(dict))
    def sort_key(item):
        (task, ntype, leaked, beta, lam), _ = item
        return (task, ntype, leaked, beta, lam if lam is not None else -1.0)

    for (task, ntype, leaked, beta, lam), losses in sorted(attacks.items(), key=sort_key):
        mse = min(losses)  # strongest attack observed for that config
        key = (leaked, lam)
        per_task[task][ntype].setdefault(beta, {})[key] = mse

    for task in TARGETS:
        if task not in bounds:
            continue
        b = bounds[task]
        print("=" * 100)
        print(f"TASK: {task.upper()}   (input_dim={b['input_dim']}, "
              f"per-coord signal power={b['per_coord_power']:.4f}  "
              f"-> zero-predictor MSE ~= {b['per_coord_power']:.3f})")
        print("=" * 100)
        header = (f"{'beta':>6} | {'bound (diag)':>13} | {'bound (full-cov)':>16} | "
                  f"{'attack MSE (aniso)':>19} | {'attack MSE (iso)':>17}")
        print(header)
        print("-" * len(header))
        betas = sorted(b["curve"].keys())
        for beta in betas:
            entry = b["curve"][beta]
            xi_diag = entry["xi_lb_diag"]
            xi_full = entry.get("xi_lb_full")
            aniso = per_task[task].get("nonisotropic", {}).get(beta, {})
            iso = per_task[task].get("isotropic", {}).get(beta, {})
            best_aniso = min(aniso.values()) if aniso else None
            best_iso = min(iso.values()) if iso else None
            ff = f"{xi_full:.6f}" if xi_full is not None else "  --  "
            fa = f"{best_aniso:.4f}" if best_aniso is not None else "  --  "
            fi = f"{best_iso:.4f}" if best_iso is not None else "  --  "
            print(f"{beta:>6.1f} | {xi_diag:>13.4f} | {ff:>16} | {fa:>19} | {fi:>17}")

        none_cfg = per_task[task].get("none", {}).get(0.0, {})
        if none_cfg:
            print(f"\n  no-noise baseline attack MSE (beta=0): best = {min(none_cfg.values()):.4f}  "
                  f"(bound is 0 when beta=0)")
        print()

    # detail: show per-config attack MSEs for the aniso case
    print("=" * 100)
    print("DETAIL: anisotropic attack MSEs per (leaked_fraction, lambda)")
    print("=" * 100)
    for task in TARGETS:
        aniso = per_task[task].get("nonisotropic", {})
        if not aniso:
            continue
        print(f"\n  {task.upper()}:")
        for beta in sorted(aniso.keys()):
            cfgs = ", ".join(
                f"leak={lk}" + (f",lam={lm}" if lm is not None else "") + f": {v:.4f}"
                for (lk, lm), v in sorted(
                    aniso[beta].items(),
                    key=lambda kv: (kv[0][0], kv[0][1] if kv[0][1] is not None else -1.0),
                )
            )
            print(f"    beta={beta:<6} {cfgs}")


if __name__ == "__main__":
    main()
