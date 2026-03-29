# Plan: Privacy-Utility Tradeoff Comparison (Your Method vs. HCR Bounds)

## Context

For the joint MTL encoder (RF + CFO + Channel only — no other configurations), we need to compare our task-aware noise mechanism (FIM-based anisotropic noise) against the HCR bounds baseline (Chaudhuri et al., Meta 2024). For the same privacy budget (`noise_level`), we measure **reconstruction hardness** (adversary MSE — higher = more private) and **downstream accuracy** (utility — higher = better). Output: JSON for later plotting.

**Note:** NeuraCrypt baseline is dropped — it doesn't work with signal data due to random patching incompatibility.

## What Already Exists (Reuse Heavily)

| Component | File | Key Functions |
|-----------|------|---------------|
| Noise injection | `code/dra_1/inject_noise.py` | `inject_isotropic_noise()`, `inject_nonisotropic_noise()` |
| FIM computation | `code/dra_1/plot_utility.py:35` | `get_empirical_fim()` |
| Utility evaluation | `code/dra_1/plot_utility.py:106-274` | `evaluate_rf/cfo/channel_utility_metric()`, `evaluate_utility_for_task()` |
| Adversary attack | `code/dra_1/robust_attack.py` | Full decoder training + MSE |
| Decoder model | `code/dra_1/models.py` | `Decoder` class |
| Model loading | `code/dra_1/model_loader.py` | `load_model_for_utility_check()`, `load_training_args()`, `create_projection_layer()`, `create_encoder()` |
| Activation dataset | `code/dra_1/py_datasets.py` | `ActivationDataset` |
| HCR reference code | `baselines/hcrbounds/codes/testers.py` | `iterate()` — power iteration via LSQR for HCR bound computation |

## Implementation Steps

### Step 1: Adapt HCR Bounds for Our Setting

**New file:** `baselines/hcrbounds/hcr_bounds.py`

The HCR paper adds isotropic Gaussian noise to features and computes a **theoretical lower bound** on the MSE of any unbiased reconstruction. The core algorithm (paper Algorithm 1, implemented in `testers.py:iterate()`) uses:
- JVP (`torch.func.jvp`) and VJP (`torch.func.vjp`) of the encoder (no full Jacobian needed)
- LSQR (scipy) power iteration to find the worst-case perturbation direction
- Bound formula: `HCR ≥ ||ε||² / (exp(||z_ε||²/σ²) - 1)` where ε is input perturbation, z_ε is feature perturbation

**Adaptations from their image code to our signal domain:**
1. Replace their `model` (ResNet/CIFAR net) with our encoder pipeline: `projection → encoder`
2. Replace image DCT domain bounds with direct signal domain MSE bounds
3. Replace their `partition()` function with our model splitting (encoder vs task heads)
4. Map our `noise_level` to their `σ`: `σ = sqrt(noise_level / latent_dim)` (per-dim variance → std)
5. Input shapes: RF (2, 1024), CFO (2, 160), Channel (2, 160) instead of images
6. Remove image-specific visualization (DCT, JPEG saving), output numerical MSE bounds

**Key functions to implement:**

```python
def build_encoder_pipeline(experiment_path, device):
    """Load trained projection+encoder as a single callable model.
    Reuse: model_loader.load_training_args(), create_projection_layer(), create_encoder()
    """

def compute_hcr_bounds(model, data_loader, sigma, device, num_iter=25, num_pits=6, diffdiv=500):
    """Adapted from testers.py iterate(). For each sample batch:
    1. Get features a_θ = model(input)
    2. Perturb features: outputsa = features + N(0, σ²)
    3. Power iteration (num_pits iters) via LSQR to find optimal ε
    4. HCR bound = ||ε||² / (exp(||z_ε||²/σ²) - 1)
    5. Max over num_iter random starts
    Returns: per-sample MSE lower bounds
    """

def compute_hcr_for_noise_levels(model, data_loader, noise_levels, latent_dim, device):
    """For each noise_level, compute σ = sqrt(noise_level/latent_dim), call compute_hcr_bounds.
    Returns: {noise_level: mean_mse_lower_bound}
    """
```

**Also create:** `baselines/__init__.py`, `baselines/hcrbounds/__init__.py`

### Step 2: Refactor `robust_attack.py` for Reuse

**Modified file:** `code/dra_1/robust_attack.py`

Extract the decoder training/eval loop from `main()` into an importable function:

```python
def run_reconstruction_attack(
    activation_dir, train_files, val_files, test_files,
    noise_type, noise_level, task, latent_dim, device, save_dir,
    L=None, V=None, lambda_factor=1e-5, clip_value=None,
    epochs=30, lr=1e-3, batch_size=64, patience=30,
    leaked_fraction=0.1
) -> dict:
    """Train adversary decoder and return reconstruction metrics.
    Returns: {'mse': float, 'nmse': float, 'correlation': float, 'snr': float}
    """
```

Keep `main()` backward compatible — it calls `run_reconstruction_attack()` internally.

### Step 3: Create Unified Comparison Script

**New file:** `compare_methods.py` (project root)

**Arguments:**
```
--experiment_path     Path to trained MTL experiment directory
--noise_levels        Noise levels to sweep (default: "0,5,10,15,20")
--output_path         Output JSON path (default: comparison_results.json)
--gpu_id              GPU ID
--leaked_fraction     Fraction for adversary (default: 0.1)
--decoder_epochs      Adversary decoder epochs (default: 30)
--fim_samples         Samples for FIM computation (default: 1000)
--lambda_factor       FIM regularization (default: 1e-5)
--skip_neuracrypt     Skip NeuraCrypt (use existing results)
--skip_hcr            Skip HCR bounds
--skip_attack         Skip adversary training (use existing results)
```

**Execution flow:**
1. Load model + activations via `model_loader`
2. Compute FIM once via `get_empirical_fim()` from `plot_utility.py`
3. **For each noise_level:**
   - **Our method (isotropic):** utility via `evaluate_utility_for_task()`, privacy via `run_reconstruction_attack()`
   - **Our method (anisotropic/FIM):** same functions with `noise_type='nonisotropic'`
   - **HCR bound:** `compute_hcr_for_noise_levels()` — pure computation, no decoder
4. Save combined JSON

**Scope:** Only the joint RF+CFO+Channel encoder (the `rf_cfo_channel` experiment directory). No single-task or partial MTL configs.

**Output JSON format:**
```json
{
  "noise_levels": [0, 5, 10, 15, 20],
  "methods": {
    "isotropic": {
      "privacy": {"rf_recon_mse": [...], "cfo_recon_mse": [...], "channel_recon_mse": [...]},
      "utility": {"rf_accuracy": [...], "cfo_mse": [...], "cfo_r2": [...], "channel_mse": [...], "channel_r2": [...]}
    },
    "anisotropic_fim": {
      "privacy": {...},
      "utility": {...}
    },
    "hcr_bound": {
      "mse_lower_bound": {"rf": [...], "cfo": [...], "channel": [...]}
    }
  },
  "metadata": {"experiment_path": "...", "latent_dim": 512, "lambda_factor": 1e-5, "timestamp": "..."}
}
```

## File Summary

| Action | File |
|--------|------|
| **New** | `baselines/__init__.py` |
| **New** | `baselines/hcrbounds/__init__.py` |
| **New** | `baselines/hcrbounds/hcr_bounds.py` — HCR bound adapted for our encoder/data |
| **Modify** | `code/dra_1/robust_attack.py` — extract `run_reconstruction_attack()` |
| **New** | `compare_methods.py` — main comparison orchestrator |

## Implementation Order

1. **Step 2** first (refactor `robust_attack.py`) — small change, prerequisite for Step 3
2. **Step 1** (HCR bounds adaptation) — can develop/test independently
3. **Step 3** (comparison script) — ties everything together

## Verification

1. Run on existing trained MTL experiment (e.g., `results_20250720_172807/rf_cfo_channel/...`)
2. Check: utility at noise_level=0 matches original eval results (sanity)
3. Check: reconstruction MSE increases with noise_level (expected behavior)
4. Check: HCR bound ≤ empirical adversary MSE at each noise_level (it's a lower bound)
5. Check: anisotropic utility ≥ isotropic utility at same noise_level (FIM-aware noise preserves more utility)
6. Verify JSON has all expected fields and no NaN/None values