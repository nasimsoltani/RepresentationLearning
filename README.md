# Representation Learning for Wireless Signal Processing

Multi-task learning (MTL) over shared representations for three wireless tasks —
**RF fingerprinting**, **channel estimation**, and **CFO (carrier frequency offset)
estimation** — plus the data reconstruction attacks and baselines used to evaluate
how much the representations leak.

## Setup

```bash
git clone https://github.com/nasimsoltani/RepresentationLearning.git
cd RepresentationLearning
uv sync                     # Python 3.11+
source .venv/bin/activate   # or prefix the commands below with `uv run`
cp .env.example .env        # then edit the paths in .env
```

`uv sync` also installs the `hf` command used below to download data and models.

### Dataset

The data is derived from the [ORACLE RF fingerprinting
dataset](https://genesys-lab.org/oracle) (16 X310 radios, `KRI-16Devices-RawData`),
preprocessed into per-packet `.mat` files.

Download it from [`Aadharsh/RepresentationLearning-dataset`](https://huggingface.co/datasets/Aadharsh/RepresentationLearning-dataset)
(3.3 GB archive plus the split file):

```bash
hf download Aadharsh/RepresentationLearning-dataset --repo-type dataset --local-dir dataset
tar xzf dataset/OracleDatasetProcessed-arranged.tar.gz -C dataset
# .env:
#   DATA_BASE_PATH=<repo>/dataset/OracleDatasetProcessed-arranged/OracleDatasetProcessed-arranged
#   PKL_FILE_PATH=<repo>/dataset
```

Note the archive has a **doubled directory nesting**; `DATA_BASE_PATH` must point at
the inner leaf directory that actually holds the `.mat` files:

```
<extract_dir>/OracleDatasetProcessed-arranged/OracleDatasetProcessed-arranged/
├── RFfingerprinting_run1_Radio<0-15>_<2-62>ft_<i>.mat   # Packet (4000x1 complex), Radio (label)
├── CFOEstimation_run1_Radio<0-15>_<2-62>ft_<i>.mat      # LSTF (160x1 complex), CFO (scalar)
└── ChannelEstimation_run1_Radio<0-15>_<2-62>ft_<i>.mat  # LLTF (160x1 complex), EstChannnel (52x1)
```

One training sample is a **triplet**: the loader takes an `RFfingerprinting_*` entry
and derives the two siblings by swapping the filename prefix, so all three families
must be present.

`rf_partition_dict_0.5.pkl` is the split index, not data — it holds the train/val/test
file lists (91,323 / 13,070 / 26,180) and the three CFO normalisation constants
(`mean_cfo`, `std_cfo`, `max_cfo`) that the released checkpoints were trained against.
Filenames in it are **relative** to `DATA_BASE_PATH`.

> Older pickles stored absolute paths. Those still load (absolute entries are used
> as-is), but they only work on the machine that generated them.

To regenerate the split from scratch instead — note this draws a *new* random
partition and recomputes the CFO constants, so it is not comparable to the released
checkpoints:

```bash
python code/preprocessing/dataset_partitioner.py   # uses DATA_BASE_PATH, PORTION_TO_USE
```

### Pretrained models

Checkpoints and the run registry: [`Aadharsh/RepresentationLearning-models`](https://huggingface.co/Aadharsh/RepresentationLearning-models).
Each run directory carries an `args.json` that `eval.py` and `inference.py` read to
rebuild the architecture, so evaluation needs no architecture flags.

## Quick start: inference with a released model

`code/rep_lr/inference.py` picks a model from the registry, downloads only its
checkpoint and `args.json` (into `models/`), and reports test metrics for every task
the model was trained on:

| Task | Metrics |
|---|---|
| RF fingerprinting | accuracy, macro-F1 |
| CFO estimation | R², RMSE / MAE / median absolute error (Hz) |
| Channel estimation | R² (mean of real and imaginary parts), NMSE (dB) |

```bash
# best RF+CFO+Channel model on the full test split
python code/rep_lr/inference.py

# best model of another category, on a reproducible 10% subset
python code/rep_lr/inference.py --task_type CFO --test_fraction 0.1 --seed 0

# list a category's candidates, or pick a run explicitly
python code/rep_lr/inference.py --task_type RF+Channel --list
python code/rep_lr/inference.py --run_id channel_estimation_20250803_153603__rf_variable
```

Categories: `RF`, `CFO`, `Channel`, `RF+CFO`, `RF+Channel`, `CFO+Channel`,
`RF+CFO+Channel`. "Best" is the run with the lowest validation loss, so the choice is
not tuned on test data. Runs trained before 2025-07-17 use an earlier encoder the
current code cannot load; automatic selection skips them.

The full test split is 26,180 packets. On a subset, CFO RMSE can differ a lot from
the registry value, because a few outlier packets account for most of the squared
error. MAE, median error and R² are more stable.

## Training

`--data_root` tells the loader where the `.mat` files live; it defaults to
`DATA_BASE_PATH` from the environment, so it can be omitted once `.env` is set.

```bash
# Single-task: RF fingerprinting
python code/rep_lr/main.py \
  --task rf_fingerprinting \
  --pkl_dataset_path "$PKL_FILE_PATH/rf_partition_dict_0.5.pkl" \
  --data_root "$DATA_BASE_PATH" \
  --epochs 300 --batch_size 1024 --gpu_id 0

# Multi-task: RF + channel + CFO
python code/rep_lr/main.py \
  --mtl --task rf_fingerprinting channel_estimation cfo_estimation \
  --pkl_dataset_path "$PKL_FILE_PATH/rf_partition_dict_0.5.pkl" \
  --data_root "$DATA_BASE_PATH" \
  --w_rf 1.0 --w_channel 1.0 --w_cfo 1.0 --gpu_id 0
```

Ready-made configurations live in `scripts/` (`train.sh`, `run_rf_fingerprinting.sh`,
`run_channel_estimation.sh`, `run_cfo_estimation_*.sh`); each sources `.env`, so no
path editing is needed.

## Evaluation

```bash
python code/rep_lr/eval.py \
  --model_path <run_dir>/..._best.pt \
  --eval_pkl_dataset_path "$PKL_FILE_PATH/rf_partition_dict_0.5.pkl" \
  --data_root "$DATA_BASE_PATH" \
  --gpu_id 0
```

`eval.py` additionally writes per-packet predictions and plots next to the checkpoint.
`--test_fraction` with `--seed` evaluates a reproducible subset.

## Privacy attack: reconstructing inputs from the representation

The threat model is a **data reconstruction attack**. An adversary who sees the
shared encoder's output `z` for a set of leaked packets (together with the packets
themselves) trains a decoder `z → x`, then tries to reconstruct unseen test packets
from their representation alone. Privacy is measured as the reconstruction **MSE**
on the test split: higher means less leakage.

Privacy noise can be added to `z` before the adversary sees it. `z` is scaled to unit
norm, Gaussian noise with total variance `--noise_level` is added, and the result is
rescaled to `min(‖z‖, --clip_value)`:

| `--noise_type` | Noise |
|---|---|
| `none` | no noise (only norm clipping, if `--clip_value` is set) |
| `isotropic` | equal variance in every latent dimension |
| `nonisotropic` | variance ∝ 1/(λᵢ + λ̄), placed away from the eigen-directions of the task heads' Fisher information (summed over tasks for multi-task models), so utility is hurt less for the same total noise |

Inputs are normalised to unit RMS, so predicting all zeros already gives an MSE of
about 0.5 on every task. An attack is only leaking information when its MSE is
clearly below that.

The attack runs in two steps on a released model.

**1. Download a model and extract its representations.**

```bash
RUN=rf_fingerprinting_cfo_estimation_channel_estimation_20250725_234643   # joint RF+CFO+Channel encoder
hf download Aadharsh/RepresentationLearning-models \
  --include "runs/$RUN/args.json" --include "runs/$RUN/*.pt" --local-dir models

python code/dra_1/extract_activations.py \
  --model_path models/runs/$RUN \
  --output_dir activations/$RUN \
  --gpu_id 0
```

This writes one `.pth` per packet (encoder output, inputs and labels, ~14 KB each):
about 2 GB for all 130,573 packets. `inference.py` stores downloaded runs in the same
`models/runs/<run_id>` layout, so a run it has already fetched can be used directly.

**2. Train the adversary and evaluate it.**

```bash
# wandb is always initialised; use offline mode unless you are logged in
WANDB_MODE=offline python code/dra_1/robust_attack.py \
  --experiment_path models/runs/$RUN \
  --activations_path activations/$RUN \
  --task channel \
  --noise_type none \
  --leaked_fraction 0.2 --clip_value 50 \
  --epochs 40 --batch_size 128 --lr 1e-3 --patience 30 \
  --latent_dim 512 \
  --output_dir attack_results/$RUN/channel/none
```

- `--task` selects which input to reconstruct: `rf` (2×1024), `cfo` or `channel` (2×160).
- `--latent_dim` must equal `2 × d2` from the run's `args.json` (512 for the released runs).
- `--leaked_fraction` is the share of the training split the adversary gets. Validation and test use the full splits.
- Add noise with `--noise_type isotropic --noise_level 10`, or
  `--noise_type nonisotropic --noise_level 10 --lambda_factor 1 --fim_samples 8000`.

The output directory contains `attack_args.json`, `epoch_history.json`,
`best_decoder.pt`, per-packet test MSEs in `attack_<task>_results.json`, and
`reconstruction_sample.png`. The final line of the log reports the test MSE. As a
reference, the command above (no noise, channel) reaches a test MSE of about 0.067
over the 26,180 test packets, against 0.500 for predicting zeros: without noise the
joint representation lets an adversary reconstruct the channel input almost exactly.

The settings above match `run_attack_exps.py`, which runs the full sweep (all tasks × `none`/`isotropic`/`nonisotropic` × noise levels 5 and 10)
in parallel with Ray. Set `results_path` at the top of the script to a directory
of trained runs first.

## Repository layout

| Path | Contents |
|---|---|
| `code/rep_lr/` | Core model (`models.py`), training (`main.py`, `train.py`), evaluation (`eval.py`), data loading (`py_datasets.py`) |
| `code/dra_1/` | Reconstruction attack (`extract_activations.py`, `robust_attack.py`), privacy noise, Fisher information, theoretical bounds |
| `code/dra/` | Earlier version of the attack code |
| `code/preprocessing/` | Raw SigMF → `.mat` conversion (`bin2mat.py`) and split generation (`dataset_partitioner.py`) |
| `code/ML/` | Legacy models and utilities |
| `baselines/` | NeuraCrypt, FIL instance encoding, RDP, HCR bounds |
| `scripts/` | Training / evaluation / experiment drivers (SLURM `job_*` files included) |

## Architecture

Three stages: a **projection layer** normalising the differing input shapes
(RF `B×2×1024`, channel/CFO `B×2×160`) to a common dimension; a shared
**convolutional encoder** producing the common representation; and per-task
**heads**. See `code/rep_lr/models.py`.

Reference architectures:
- Channel estimator (page 3, first column): https://genesys-lab.org/papers/SPiNN.pdf
- CFO estimator (figure 7a): https://par.nsf.gov/servlets/purl/10487629
- RF fingerprinting: ?

## Citation

If you use this code or the derived dataset, please cite this repository. The ORACLE authors ask that any publication using their data
also cite:

```bibtex
@inproceedings{sankhe2019oracle,
  title={ORACLE: Optimized Radio clAssification through Convolutional neuraL nEtworks},
  author={Sankhe, Kunal and Belgiovine, Mauro and Zhou, Fan and Riyaz, Shamnaz and Ioannidis, Stratis and Chowdhury, Kaushik},
  booktitle={IEEE INFOCOM 2019 - IEEE Conference on Computer Communications},
  pages={370--378},
  year={2019}
}
```
