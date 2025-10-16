import os
import ray
import subprocess
import time
import dotenv
import glob
import json
import pandas as pd
from typing import List, Dict, Any

dotenv.load_dotenv()

# Explicitly capture environment variables from .env to pass to Ray workers
env_vars_from_dotenv = dotenv.dotenv_values()
# Ray's runtime_env expects string values, so filter out any Nones
safe_env_vars = {k: v for k, v in env_vars_from_dotenv.items() if v is not None}

# Define noise levels and lambda factors (same as in run_attack_exps.py)
NOISE_LEVELS = [5, 10, 15, 20]
LAMBDA_FACTORS = [1e+1,1,1e-1,1e-2,1e-3, 1e-4 ]#[1e-1,1e-2,1e-3] 
# lambda_factors = [0.01]
NOISE_TYPES = ["none", "isotropic", "nonisotropic"]

# Mapping for encoder_train_task
TASK_MAPPING = {
    "rf": "rf_fingerprinting", 
    "cfo": "cfo_estimation", 
    "channel": "channel_estimation"
}

@ray.remote(num_gpus=0.01)
def run_command(command: str, task_type: str):
    """
    This worker function now only 'reserves' 0.1 of a GPU's compute.
    Ray can now schedule 10 of these to run on a single GPU simultaneously.
    """
    print(f"🚀 Starting {task_type} task: {command}")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Finished {task_type} task: {command}")
    else:
        print(f"❌ ERROR in {task_type} task: {command}\n   STDERR: {result.stderr}")

    return result.returncode

# --- Configuration ---
# You can set these paths in your .env file or modify them here.
results_path = "/scratch/10608/aadharsh_aadhithya/repos/RepresentationLearning/results_parallel_20250829_214158/rf_fixed"
activations_base = os.getenv("ACTIVATIONS_BASE", "/work/10608/aadharsh_aadhithya/vista/RepresentationLearning/activations")
utility_script = "code/dra_1/plot_utility_lambda.py"
#ray_tmp_dir = os.getenv("RAY_TMP_DIR")

# This flag determines how the activation path is constructed.
# Set to True if activations are stored in a central location (`activations_base`).
# Set to False if they are stored within each experiment directory.
REMOTE_ACTIVATION = True 

# if not os.path.exists(ray_tmp_dir):
#     os.makedirs(ray_tmp_dir, exist_ok=True)


def parse_encoder_train_task(task_dir_name: str) -> List[str]:
    """
    Parse the task directory name and convert to encoder_train_task list.
    
    Args:
        task_dir_name: Directory name like 'cfo', 'rf_cfo_channel', etc.
    
    Returns:
        List of task names using the mapping
    """
    parts = task_dir_name.split('_')
    encoder_tasks = []
    
    for part in parts:
        if part in TASK_MAPPING:
            encoder_tasks.append(TASK_MAPPING[part])
    
    return encoder_tasks





def generate_utility_command(experiment_path, activations_path, output_dir, noise_type, noise_level, lambda_factor=None):
    """Generates the command to run the plot_utility_lambda.py script."""
    
    cmd = (f"python {utility_script} "
           f"--experiment_path {experiment_path} "
           f"--activation_dir {activations_path} "
           f"--output_dir {output_dir} "
           f"--noise_type {noise_type} "
           f"--noise_level {noise_level}")
    
    if lambda_factor is not None and noise_type == 'nonisotropic':
        cmd += f" --lambda_factor {lambda_factor}"
           
    return cmd


def collect_utility_results(experiment_path: str) -> List[Dict[str, Any]]:
    """
    Collect utility results from JSON files in the experiment directory.
    
    Returns:
        List of dictionaries containing utility metrics
    """
    results = []
    utility_plots_dir = os.path.join(experiment_path, 'utility_plots_lambda')
    
    if not os.path.exists(utility_plots_dir):
        print(f"Warning: Utility plots directory not found: {utility_plots_dir}")
        return results
    
    # Look for JSON result files
    json_pattern = os.path.join(utility_plots_dir, '**', '*.json')
    
    for json_file in glob.glob(json_pattern, recursive=True):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract path components to determine noise_type, noise_level, lambda_factor
            rel_path = os.path.relpath(json_file, utility_plots_dir)
            path_parts = rel_path.split(os.sep)
            
            # Parse path structure: noise_type/noise_level_X_Y/[lambda_factor]/task/results.json
            if len(path_parts) >= 3:
                noise_type = path_parts[0]
                noise_level_str = path_parts[1]
                
                # Extract noise level
                try:
                    noise_level = float(noise_level_str.split('level_')[-1].replace('_', '.'))
                except:
                    noise_level = 0.0
                
                # Check for lambda factor
                lambda_factor = None
                task_idx = 2
                if len(path_parts) > 3 and path_parts[2].startswith('lambda_'):
                    lambda_factor_str = path_parts[2].split('lambda_')[-1]
                    lambda_factor = float(lambda_factor_str.replace('_', '.').replace('-', '-'))
                    task_idx = 3
                
                # Extract task
                if len(path_parts) > task_idx:
                    task = path_parts[task_idx]
                else:
                    task = 'unknown'
                
                # Add to results
                result_entry = {
                    'noise_type': noise_type,
                    'noise_level': noise_level,
                    'lambda_factor': lambda_factor,
                    'task': task,
                    **data  # Include all metrics from JSON
                }
                results.append(result_entry)
                
        except Exception as e:
            print(f"Error reading JSON file {json_file}: {e}")
            continue
    
    return results


def main():
    utility_commands = []
    all_results = []

    # Get all task directories from the main results folder
    try:
        tasks = os.listdir(results_path)
    except FileNotFoundError:
        print(f"Error: Results path not found at '{results_path}'. Please check the path.")
        return

    for task_dir_name in tasks:
        # Construct the full path to the specific task-run directory
        task_path = os.path.join(results_path, task_dir_name)
        if not os.path.isdir(task_path):
            continue
            
        # Find the timestamped experiment folder inside
        try:
            #there might be files also so we need to check if it is a directory
            experiment_folder = None 
            for folder in os.listdir(task_path):
                if os.path.isdir(os.path.join(task_path, folder)):
                    experiment_folder = folder
                    break
            
            if experiment_folder is None:
                print(f"Cannot find experiment fodler. Skipping empty directory: {task_path}")
                continue
                
            full_experiment_path = os.path.join(task_path, experiment_folder)

            
        except StopIteration:
            print(f"Skipping empty directory: {task_path}")
            continue

        print(f"\n--- Processing Experiment: {full_experiment_path} ---")

        # Determine the path to the stored activations
        if REMOTE_ACTIVATION:
            results_name = os.path.basename(results_path)
            activations_path = os.path.join(activations_base, results_name, task_dir_name)
        else:
            activations_path = os.path.join(full_experiment_path, "activations")

        print(f"Using activations path: {activations_path}")
        if not os.path.exists(activations_path):
            print(f"Warning: Activations path does not exist, skipping: {activations_path}")
            continue

        # Find the directory containing attack results
        # attack_results_dir = os.path.join(full_experiment_path, 'attack_results_robust')
        
        # if not os.path.isdir(attack_results_dir):
        #     print(f"Warning: 'attack_results_robust' directory not found in {full_experiment_path}.")
        #     continue
        
        # Get encoder_train_task for this experiment
        encoder_train_task = parse_encoder_train_task(task_dir_name)
        
        # Define where to save the utility plots and logs
        output_plot_dir = os.path.join(full_experiment_path, 'utility_plots_lambda')
        os.makedirs(output_plot_dir, exist_ok=True)

        # Generate commands for each combination
        for noise_type in NOISE_TYPES:
            if noise_type == "none":
                noise_levels_to_use = [0]
            else:
                noise_levels_to_use = NOISE_LEVELS
            
            for noise_level in noise_levels_to_use:
                if noise_type == 'nonisotropic':
                    # Include lambda factor sweep for nonisotropic noise
                    for lambda_factor in LAMBDA_FACTORS:
                        # Create specific output directory
                        lambda_str = str(lambda_factor).replace('.', '_').replace('-', 'neg')
                        level_str = str(float(noise_level)).replace('.', '_')
                        specific_output_dir = os.path.join(
                            output_plot_dir, 
                            noise_type, 
                            f'noise_level_{level_str}',
                            f'lambda_{lambda_str}'
                        )
                        os.makedirs(specific_output_dir, exist_ok=True)
                        
                        # Generate command
                        command = generate_utility_command(
                            experiment_path=full_experiment_path,
                            activations_path=activations_path,
                            output_dir=specific_output_dir,
                            noise_type=noise_type,
                            noise_level=noise_level,
                            lambda_factor=lambda_factor
                        )
                        
                        # Redirect output to a log file
                        log_file_path = os.path.join(specific_output_dir, 'utility_run.log')
                        command_with_log = f"{command} > {log_file_path} 2>&1"
                        utility_commands.append(command_with_log)
                else:
                    # For isotropic and none, no lambda factor
                    level_str = str(float(noise_level)).replace('.', '_')
                    specific_output_dir = os.path.join(
                        output_plot_dir, 
                        noise_type, 
                        f'noise_level_{level_str}'
                    )
                    os.makedirs(specific_output_dir, exist_ok=True)
                    
                    # Generate command
                    command = generate_utility_command(
                        experiment_path=full_experiment_path,
                        activations_path=activations_path,
                        output_dir=specific_output_dir,
                        noise_type=noise_type,
                        noise_level=noise_level
                    )
                    
                    # Redirect output to a log file
                    log_file_path = os.path.join(specific_output_dir, 'utility_run.log')
                    command_with_log = f"{command} > {log_file_path} 2>&1"
                    utility_commands.append(command_with_log)

    print(f"\nGenerated {len(utility_commands)} utility plot commands.")
    if not utility_commands:
        print("No commands to run. Exiting.")
        return

    # --- Initialize Ray and run the commands ---
    ray.init(
        #_temp_dir=ray_tmp_dir,
        runtime_env={
            "conda": "vllm", # Make sure this matches your conda environment name
            "env_vars": safe_env_vars,
        }
    )
    print(f"\nRay cluster started. Available resources: {ray.available_resources()}")

    print("\n" + "="*50 + "\nSubmitting utility plot generation tasks.")
    utility_futures = [run_command.remote(cmd, "utility_plot") for cmd in utility_commands]
    ray.get(utility_futures)
    print("\n🎉 All utility plots have been generated!")

    # --- Collect results and generate CSV ---
    print("\n" + "="*50 + "\nCollecting results and generating CSV.")
    
    try:
        tasks = os.listdir(results_path)
    except FileNotFoundError:
        print(f"Error: Results path not found at '{results_path}'. Please check the path.")
        return

    for task_dir_name in tasks:
        task_path = os.path.join(results_path, task_dir_name)
        if not os.path.isdir(task_path):
            continue
            
        try:
            experiment_folder = None 
            for folder in os.listdir(task_path):
                if os.path.isdir(os.path.join(task_path, folder)):
                    experiment_folder = folder
                    break
            
            if experiment_folder is None:
                print(f"Cannot find experiment fodler. Skipping empty directory: {task_path}")
                continue
                
            full_experiment_path = os.path.join(task_path, experiment_folder)
            # experiment_folder = os.listdir(task_path)[0]
            # full_experiment_path = os.path.join(task_path, experiment_folder)
        except IndexError:
            continue

        # Get encoder_train_task for this experiment
        encoder_train_task = parse_encoder_train_task(task_dir_name)
        
        # Collect utility results from this experiment
        experiment_results = collect_utility_results(full_experiment_path)
        
        # Add encoder_train_task to each result
        for result in experiment_results:
            result['encoder_train_task'] = str(encoder_train_task)  # Convert to string for CSV
        
        all_results.extend(experiment_results)

    # Create DataFrame and save CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Ensure we have the required columns
        required_columns = ['noise_type', 'noise_level', 'lambda_factor', 'encoder_train_task', 'task', 'r2', 'mse', 'accuracy']
        
        # Add missing columns with default values if they don't exist
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Reorder columns
        df = df[required_columns + [col for col in df.columns if col not in required_columns]]
        
        # Save CSV
        csv_path = os.path.join(results_path, 'utility_results_lambda_sweep.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n📊 CSV file saved: {csv_path}")
        print(f"Total results collected: {len(all_results)}")
    else:
        print("\n⚠️  No results collected for CSV generation.")


if __name__ == "__main__":
    main()