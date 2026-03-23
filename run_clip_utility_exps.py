import os
import ray
import subprocess
import dotenv
import glob
import json
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any

dotenv.load_dotenv()

# Explicitly capture environment variables from .env to pass to Ray workers
env_vars_from_dotenv = dotenv.dotenv_values()
safe_env_vars = {k: v for k, v in env_vars_from_dotenv.items() if v is not None}

# Configuration
LAMBDA_FACTORS = [0.001, 0.01, 0.1, 1.0]
BETA_ISO = 5.0
LATENT_DIM = 512  # Assumed d
LAMBDA_MAX = 1.0  # Assumed lambda_max

NOISE_TYPES = ["none", "isotropic", "nonisotropic"]

ALLOWED_TASK_DIRS = {
    "rf_cfo_channel"
}

# Sweep over clip valu1500es
CLIP_VALUES = [1500]

UTILITY_SCRIPT = "code/dra_1/plot_utility_lambda.py"


def calculate_k_factor(d, lambda_max, gamma):
    """
    Calculates K based on the approximation formula:
    K ≈ (1/d) * [1 + ((d-1)*(lambda_max + gamma)) / gamma]
    """
    if gamma == 0: return float('inf')
    term2 = ((d - 1) * (lambda_max + gamma)) / gamma
    k = (1.0 / d) * (1.0 + term2)
    return k


@ray.remote(num_gpus=0.01)
def run_command(command: str, task_type: str):
    print(f"🚀 Starting {task_type} task: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Finished {task_type} task: {command}")
    else:
        print(f"❌ ERROR in {task_type} task: {command}\n   STDERR: {result.stderr}")
    return result.returncode


def generate_utility_command(
    experiment_path: str,
    activations_path: str,
    output_dir: str,
    noise_type: str,
    noise_level: float,
    lambda_factor: float | None = None,
    clip_value: float | None = None,
) -> str:

    cmd = (
        f"python {UTILITY_SCRIPT} "
        f"--experiment_path {experiment_path} "
        f"--activation_dir {activations_path} "
        f"--output_dir {output_dir} "
        f"--noise_type {noise_type} "
        f"--noise_level {noise_level}"
    )

    if lambda_factor is not None and noise_type == "nonisotropic":
        cmd += f" --lambda_factor {lambda_factor}"

    if clip_value is not None:
        cmd += f" --clip_value {clip_value}"

    return cmd


def collect_clip_utility_results(experiment_path: str) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []
    utility_plots_dir = os.path.join(experiment_path, "utility_plots_clip")

    if not os.path.exists(utility_plots_dir):
        print(f"Warning: Utility plots directory not found: {utility_plots_dir}")
        return results

    json_pattern = os.path.join(utility_plots_dir, "**", "*.json")

    for json_file in glob.glob(json_pattern, recursive=True):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            rel_path = os.path.relpath(json_file, utility_plots_dir)
            path_parts = rel_path.split(os.sep)

            if len(path_parts) >= 3:
                noise_type = path_parts[0]
                noise_level_str = path_parts[1]

                try:
                    noise_level = float(noise_level_str.split("noise_level_")[-1].replace("_", "."))
                except Exception:
                    noise_level = 0.0

                clip_value = None
                lambda_val = None
                k_val = None
                
                # Parse path components to extract params
                # Expected structure variants:
                # 1. noise_type/noise_level_X/task/results.json
                # 2. noise_type/noise_level_X/clip_Y/task/results.json
                # 3. noise_type/noise_level_X/lambda_Z/clip_Y/task/results.json
                
                # We need to robustly find the task name (last directory)
                task = path_parts[-2]
                
                # Check for clip and lambda in intermediate folders
                for part in path_parts[2:-2]: # Skip noise_type, noise_level, and task/filename
                    if part.startswith("clip_"):
                        try:
                            clip_str = part.split("clip_")[-1]
                            if clip_str == "noclip":
                                clip_value = None
                            else:
                                clip_value = float(clip_str.replace("_", "."))
                        except: pass
                    elif part.startswith("lambda_"):
                        try:
                            lambda_str = part.split("lambda_")[-1]
                            lambda_val = float(lambda_str.replace("_", "."))
                        except: pass

                # If lambda found, calculate K
                if lambda_val is not None:
                    k_val = calculate_k_factor(LATENT_DIM, LAMBDA_MAX, lambda_val)

                results.append({
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                    "clip_value": clip_value,
                    "lambda_factor": lambda_val,
                    "K_factor": k_val,
                    "task": task,
                    **data,
                })

        except Exception as e:
            print(f"Error reading JSON file {json_file}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run utility evaluations over a sweep of clip values and calculated beta_noniso."
    )

    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to main experiment directory.",
    )
    parser.add_argument(
        "--activation_dir",
        type=str,
        default=None,
        help="Optional base path for activations.",
    )
    # Removed single lambda_factor arg since we now iterate over a list

    args = parser.parse_args()

    experiment_path = args.experiment_path
    if not os.path.isdir(experiment_path):
        print(f"Error: experiment_path is not a directory: {experiment_path}")
        return

    run_name = os.path.basename(experiment_path.rstrip(os.sep))
    activations_base = args.activation_dir or os.getenv("ACTIVATIONS_BASE")

    try:
        task_dirs = [
            d for d in os.listdir(experiment_path)
            if d in ALLOWED_TASK_DIRS and os.path.isdir(os.path.join(experiment_path, d))
        ]
    except FileNotFoundError:
        print(f"Error: experiment_path not found at '{experiment_path}'.")
        return

    print(f"\nUsing task directories: {task_dirs}")

    utility_commands: List[str] = []

    for task_dir_name in task_dirs:
        task_path = os.path.join(experiment_path, task_dir_name)
        
        # Find experiment folder
        experiment_folder = None
        for folder in os.listdir(task_path):
            if os.path.isdir(os.path.join(task_path, folder)):
                experiment_folder = folder
                break
        if experiment_folder is None: continue

        full_experiment_path = os.path.join(task_path, experiment_folder)
        print(f"\n--- Processing: {full_experiment_path} ---")

        if activations_base is not None:
            activations_path = os.path.join(activations_base, run_name, task_dir_name)
        else:
            activations_path = os.path.join(full_experiment_path, "activations")

        if not os.path.exists(activations_path):
            print(f"Warning: Activations path missing: {activations_path}")
            continue

        output_root_dir = os.path.join(full_experiment_path, "utility_plots_clip")
        os.makedirs(output_root_dir, exist_ok=True)

        for noise_type in NOISE_TYPES:
            
            # Determine configurations to run based on noise type
            configs_to_run = [] # List of (noise_level, lambda_factor) tuples

            if noise_type == "none":
                configs_to_run.append((0.0, None))
            
            elif noise_type == "isotropic":
                # Just run the base level
                configs_to_run.append((BETA_ISO, None))
            
            elif noise_type == "nonisotropic":
                # Run for each lambda factor
                for gamma in LAMBDA_FACTORS:
                    # Calculate K and beta_noniso
                    k = calculate_k_factor(LATENT_DIM, LAMBDA_MAX, gamma)
                    beta_noniso = k * BETA_ISO
                    
                    # Add both calculated beta and base beta
                    configs_to_run.append((beta_noniso, gamma))
                    configs_to_run.append((BETA_ISO, gamma))

            # Generate commands for these configs
            for noise_level, lambda_val in configs_to_run:
                
                # For "none", run once without clip loop logic (handled inside utility script effectively)
                if noise_type == "none":
                     level_str = str(float(noise_level)).replace(".", "_")
                     # "None" noise type usually ignores clip value in directory structure unless we force it
                     # But to keep consistent with previous script, let's sweep clips for "none" too if needed,
                     # or just run once. The previous script ran clip sweep for "none".
                     
                     for clip_value in CLIP_VALUES:
                        clip_label = "noclip" if clip_value is None else str(clip_value).replace(".", "_")
                        specific_output_dir = os.path.join(
                            output_root_dir, noise_type, f"noise_level_{level_str}", f"clip_{clip_label}"
                        )
                        os.makedirs(specific_output_dir, exist_ok=True)
                        
                        cmd = generate_utility_command(
                            full_experiment_path, activations_path, specific_output_dir,
                            noise_type, noise_level, clip_value=clip_value
                        )
                        utility_commands.append(f"{cmd} > {os.path.join(specific_output_dir, 'run.log')} 2>&1")
                     continue

                # For noisy cases
                for clip_value in CLIP_VALUES:
                    level_str = str(float(noise_level)).replace(".", "_")
                    clip_label = "noclip" if clip_value is None else str(clip_value).replace(".", "_")
                    
                    # Construct path: .../noise_type/noise_level/[lambda]/clip/
                    path_components = [output_root_dir, noise_type, f"noise_level_{level_str}"]
                    if lambda_val is not None:
                        path_components.append(f"lambda_{str(lambda_val).replace('.', '_')}")
                    path_components.append(f"clip_{clip_label}")
                    
                    specific_output_dir = os.path.join(*path_components)
                    os.makedirs(specific_output_dir, exist_ok=True)

                    cmd = generate_utility_command(
                        full_experiment_path, activations_path, specific_output_dir,
                        noise_type, noise_level, lambda_factor=lambda_val, clip_value=clip_value
                    )
                    utility_commands.append(f"{cmd} > {os.path.join(specific_output_dir, 'run.log')} 2>&1")

    print(f"\nGenerated {len(utility_commands)} commands.")
    if not utility_commands: return

    ray.init(address="local", runtime_env={"conda": "vllm", "env_vars": safe_env_vars})
    print(f"Ray resources: {ray.available_resources()}")

    futures = [run_command.remote(cmd, "utility_clip") for cmd in utility_commands]
    ray.get(futures)
    print("\n🎉 All utility plots generated!")

    # Collect results
    print("\nCollecting results...")
    all_results = []
    
    for task_dir_name in task_dirs:
        # ... (same directory finding logic)
        task_path = os.path.join(experiment_path, task_dir_name)
        experiment_folder = None
        for folder in os.listdir(task_path):
            if os.path.isdir(os.path.join(task_path, folder)):
                experiment_folder = folder
                break
        if experiment_folder is None: continue
        
        full_experiment_path = os.path.join(task_path, experiment_folder)
        all_results.extend(collect_clip_utility_results(full_experiment_path))

    if all_results:
        df = pd.DataFrame(all_results)
        required_columns = ["noise_type", "noise_level", "clip_value", "lambda_factor", "K_factor", "task", "r2", "mse", "accuracy"]
        for col in required_columns:
            if col not in df.columns: df[col] = None
        
        df = df[required_columns + [c for c in df.columns if c not in required_columns]]
        csv_path = os.path.join(experiment_path, "utility_results_beta_scaled.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📊 CSV saved: {csv_path}")
        print(f"Total results: {len(all_results)}")
    else:
        print("\n⚠️ No results found.")

if __name__ == "__main__":
    main()
