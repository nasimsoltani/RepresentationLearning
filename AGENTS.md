# Repository Guidelines

## Project Structure & Module Organization
Core research code lives under `code/`. The main active packages are `code/rep_lr/` for representation-learning experiments, `code/dra/` and `code/dra_1/` for attack and activation workflows, and `code/ML/` for older model training and evaluation utilities. Top-level scripts such as `run_attack_exps.py`, `run_utility_exps.py`, and `run_parallel_experiments.py` orchestrate larger experiment batches. Input data and derived artifacts typically live in `dataset/`, `results*/`, and `wandb/`. Treat `baselines/` as imported comparison code: avoid broad refactors there unless you are changing a specific baseline.

## Build, Test, and Development Commands
Use Python 3.11+ and `uv` for local execution.

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python main.py`: smoke-test the project entry point.
- `uv run python code/rep_lr/main.py --help`: inspect training options for the main pipeline.
- `bash code/run_training.sh`: run the repository’s saved training entry script.
- `uv run python run_attack_exps.py`: launch Ray-based attack experiments.

Prefer `uv run python <script>.py` for ad hoc experiments so dependency resolution stays consistent.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation in active code, `snake_case` for functions, variables, and files, and `CamelCase` for classes. Keep new modules focused and script names descriptive, for example `run_specific_attacks.py`. There is no repository-wide formatter or linter configured in `pyproject.toml`, so keep imports tidy, avoid dead code, and match surrounding style in the file you touch.

## Testing Guidelines
Test coverage is lightweight and script-driven rather than centralized. Existing checks include `code/dra/test_py_datasets.py` and `code/ML/test_model.py`. Run targeted validation with `uv run python code/dra/test_py_datasets.py` or the specific experiment script you changed. For new work, add narrow reproducible checks near the affected module and name test files `test_<feature>.py`.

## Commit & Pull Request Guidelines
Recent commits use short, pragmatic subjects such as `rdp corrected` and `neuracrypt fixed`. Keep commit messages brief, imperative, and scoped to one change. Pull requests should state the experiment or module affected, list the commands you ran, note dataset or `.env` assumptions, and include plots or metrics when behavior changes are empirical.

## Configuration Tips
This repo relies on local absolute paths and environment variables from `.env`. Avoid hard-coding new machine-specific paths; prefer environment-backed configuration and document any required variables in the PR.
