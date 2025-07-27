import os
import ray
import subprocess
import time
import dotenv
import glob

dotenv.load_dotenv()

# Explicitly capture environment variables from .env to pass to Ray workers
env_vars_from_dotenv = dotenv.dotenv_values()
# Ray's runtime_env expects string values, so filter out any Nones
safe_env_vars = {k: v for k, v in env_vars_from_dotenv.items() if v is not None}


@ray.remote(num_gpus=0.1)
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
results_path = "/work/10608/aadharsh_aadhithya/vista/RepresentationLearning/results_20250723_164619"
activations_base = os.getenv("ACTIVATIONS_BASE", "/work/10608/aadharsh_aadhithya/vista/RepresentationLearning/activations")
utility_script = os.getenv("UTILITY_SCRIPT", "code/dra_1/plot_utility.py")
ray_tmp_dir = os.getenv("RAY_TMP_DIR", "/tmp/ray")

# This flag determines how the activation path is constructed.
# Set to True if activations are stored in a central location (`activations_base`).
# Set to False if they are stored within each experiment directory.
REMOTE_ACTIVATION = True 

if not os.path.exists(ray_tmp_dir):
    os.makedirs(ray_tmp_dir, exist_ok=True)


def find_noise_levels(base_path: str) -> list[float]:
    """Scans subdirectories to find all unique noise levels used in attacks."""
    noise_levels = set()
    print(f"Scanning for noise levels in: {base_path}")
    
    # We look for directories matching the pattern `.../noise_level_X_Y/...`
    # The glob pattern `**/` will search recursively.
    search_pattern = os.path.join(base_path, '**', '*level_*')
    
    for path in glob.glob(search_pattern, recursive=True):
        if os.path.isdir(path):
            dirname = os.path.basename(path)
            try:
                # e.g., "noise_level_1_5" -> "1_5" -> "1.5"
                level_str = dirname.split('level_')[-1]
                level_float = float(level_str.replace('_', '.'))
                noise_levels.add(level_float)
            except (IndexError, ValueError):
                print(f"Could not parse noise level from directory: {dirname}")
                continue
                
    sorted_levels = sorted(list(noise_levels))
    print(f"Found noise levels: {sorted_levels}")
    return sorted_levels


def generate_utility_command(experiment_path, activations_path, output_dir, noise_levels_list):
    """Generates the command to run the plot_utility.py script."""
    noise_levels_str = ",".join(map(str, noise_levels_list))
    
    cmd = (f"python {utility_script} "
           f"--experiment_path {experiment_path} "
           f"--activation_dir {activations_path} "
           f"--output_dir {output_dir} "
           f"--noise_levels \"{noise_levels_str}\"")
           
    return cmd


def main():
    utility_commands = []

    # Get all task directories from the main results folder
    try:
        tasks = os.listdir(results_path)
    except FileNotFoundError:
        print(f"Error: Results path not found at '{results_path}'. Please check the path.")
        return

    for task_dir_name in tasks:
        # Construct the full path to the specific task-run directory
        # e.g., .../results_20250723_164619/rf_cfo
        task_path = os.path.join(results_path, task_dir_name)
        if not os.path.isdir(task_path):
            continue
            
        # Find the timestamped experiment folder inside, e.g., rf_cfo_channel_20250724_120000
        try:
            experiment_folder = os.listdir(task_path)[0]
            full_experiment_path = os.path.join(task_path, experiment_folder)
        except IndexError:
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

        # Find the directory containing attack results to scan for noise levels
        attack_results_dir = os.path.join(full_experiment_path, 'attack_results_robust')
        
        if not os.path.isdir(attack_results_dir):
            print(f"Warning: 'attack_results_robust' directory not found in {full_experiment_path}. Cannot determine noise levels.")
            continue
        
        # Discover noise levels by scanning the attack results directory
        noise_levels = find_noise_levels(attack_results_dir)
        if not noise_levels:
            print(f"Warning: No noise levels found for experiment {full_experiment_path}. Skipping.")
            continue

        # Define where to save the utility plots and logs
        output_plot_dir = os.path.join(full_experiment_path, 'utility_plots')
        os.makedirs(output_plot_dir, exist_ok=True)

        # Generate the command for plot_utility.py
        command = generate_utility_command(
            experiment_path=full_experiment_path,
            activations_path=activations_path,
            output_dir=output_plot_dir,
            noise_levels_list=noise_levels
        )

        # Redirect output to a log file
        log_file_path = os.path.join(output_plot_dir, 'utility_run.log')
        command_with_log = f"{command} > {log_file_path} 2>&1"
        utility_commands.append(command_with_log)

    print(f"\nGenerated {len(utility_commands)} utility plot commands.")
    if not utility_commands:
        print("No commands to run. Exiting.")
        return

    # --- Initialize Ray and run the commands ---
    ray.init(
        _temp_dir=ray_tmp_dir,
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


if __name__ == "__main__":
    main() 