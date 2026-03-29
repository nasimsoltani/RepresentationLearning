# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a representation learning project that implements multi-task learning (MTL) for wireless signal processing tasks. The codebase trains shared neural network representations that can simultaneously perform RF fingerprinting, channel estimation, and CFO (Carrier Frequency Offset) estimation.

## Architecture

The project is organized into several key modules:

### Core Representation Learning (`code/rep_lr/`)
- **main.py**: Entry point for training models with configurable tasks, architecture, and hyperparameters.
- **models.py**: Defines neural network architecture with three main components:
  1. **Projection layer**: Maps different input shapes (RF: B×2×1024, Channel/CFO: B×2×160) to a common dimension
  2. **Encoder**: Extracts common representation across all tasks using convolutional blocks
  3. **Task heads**: Specialized output heads for RF fingerprinting, channel estimation, and CFO estimation
- **train.py**: Training loop with early stopping, checkpointing, and WandB integration
- **eval.py**: Evaluation and attack evaluation (inference-time membership inference attacks)
- **py_datasets.py**: Dataset loading and preprocessing

### Attack/Analysis Modules
- **code/dra/**: Data Representation Analysis (includes activation extraction, privacy attacks)
  - `extract_activations.py`: Extracts intermediate layer activations from trained models
  - `attack.py`: Implements membership inference attacks
  - `test_py_datasets.py`: Unit tests for dataset loading with various noise configurations
- **code/dra_1/**: Earlier variant with additional analysis tools like eigenspectrum analysis

### Supporting Modules
- **code/preprocessing/**: Dataset preparation and file management scripts
- **code/ML/**: General ML utilities and legacy models

## Common Commands

### Training a Model
```bash
# Single-task RF fingerprinting training
python code/rep_lr/main.py \
  --task rf_fingerprinting \
  --pkl_dataset_path dataset/rf_partition_dict_0.5.pkl \
  --gpu_id 0 \
  --epochs 300 \
  --batch_size 1024

# Multi-task learning (RF + Channel + CFO)
python code/rep_lr/main.py \
  --task rf_fingerprinting channel_estimation cfo_estimation \
  --mtl \
  --pkl_dataset_path dataset/rf_partition_dict_0.5.pkl \
  --gpu_id 0
```

### Key Training Arguments
- `--task`: One or more tasks from `rf_fingerprinting`, `channel_estimation`, `cfo_estimation`
- `--mtl`: Enable multi-task learning (required if multiple tasks specified)
- `--pkl_dataset_path`: Path to dataset pickle file
- `--slice_len`: Input length (1024 for RF, 160 for CFO/Channel)
- `--epochs`, `--batch_size`, `--lr`: Standard hyperparameters
- `--encoder_num_blocks`: Number of convolutional blocks in encoder (1 for simple, 3-5 for complex)
- `--task_adaptive_encoder`: Use task-adaptive encoder for better MTL performance
- `--adaptive_cfo`, `--direct_cfo`, `--simple_cfo`: Special CFO head variants
- `--save_path`: Directory to save results and checkpoints

### Testing
```bash
# Test dataset loading with various noise configurations
python code/dra/test_py_datasets.py \
  --activations_dir <path_to_activations> \
  --batch_size 32 \
  --noise_type gaussian \
  --noise_level 0.1
```

### Extracting Model Activations
```bash
python code/dra/extract_activations.py \
  --model_path <path_to_checkpoint> \
  --dataset_path <path_to_data> \
  --output_dir activations/
```

### Evaluating Model and Running Attacks
```bash
# Single-task evaluation
python code/rep_lr/eval.py \
  --ckpt_path <checkpoint_path> \
  --pkl_dataset_path dataset/rf_partition_dict_0.5.pkl \
  --gpu_id 0

# Multi-task evaluation
python code/rep_lr/eval.py \
  --ckpt_path <checkpoint_path> \
  --pkl_dataset_path dataset/rf_partition_dict_0.5.pkl \
  --mtl \
  --gpu_id 0
```

## Key Design Patterns

### Input Normalization
The projection layers handle input shape variation:
- RF data comes as (B, 2, 1024) - raw RF samples with I/Q components
- Channel/CFO data comes as (B, 2, 160) - shorter sequences
- `UpsamplingProjector` or `ComplexSequenceProjector` normalizes to common dimension before encoder

### Multi-Task Learning (MTL)
- Tasks share a common encoder but have separate task heads
- Loss weighting via `--w_rf`, `--w_channel`, `--w_cfo` parameters
- Fusion types: `sum` (element-wise), `concat`, `depth_concat`
- Task-adaptive encoder can learn task-specific feature importance

### Dataset Structure
Datasets are loaded from pickle files containing:
- Training/validation/test splits
- RF samples (raw I/Q sequences)
- Channel estimates
- CFO values
- RF fingerprinting labels (device IDs)

## Development Notes

- **Environment**: Python 3.11+, uses `uv` for dependency management
- **Dependencies**: torch, ray (for parallel experiments), scikit-learn, scipy, seaborn, tqdm, dotenv
- **Logging**: Integrates with WandB for experiment tracking
- **Device**: Specify GPU via `--gpu_id` (default: 0)
- **Results**: All outputs go to `--save_path` (default: `results/`), organized by task and timestamp
- **Checkpointing**: Models saved every `--save_epochs` epochs (default: 10), enables resuming with `--resume_from`
- **Early Stopping**: Patience-based early stopping (default: 10 epochs) based on validation loss

## Experiment Scripts

Several utility scripts for running experiments in parallel:
- `run_parallel_experiments.py`: Orchestrates multiple training runs
- `run_attack_exps.py`: Runs membership inference attacks across models
- `run_utility_exps.py`: Evaluates model utility (accuracy) under various conditions
- `generate_line_plots.py`: Generates result visualizations

Job submission scripts (`job_*` files) use SLURM for HPC cluster execution.
