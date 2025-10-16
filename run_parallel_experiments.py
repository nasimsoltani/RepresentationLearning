import os
import ray
import subprocess
import time
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
env_vars_from_dotenv = dotenv.dotenv_values()
# Ray's runtime_env expects string values, so filter out any Nones
safe_env_vars = {k: v for k, v in env_vars_from_dotenv.items() if v is not None}


@ray.remote(num_gpus=0.04)
def run_command(command: str, task_name: str):
    """
    This worker function reserves 1 full GPU for an experiment.
    Ray's scheduler will manage placing these tasks on available GPUs.
    """
    print(f"🚀 Starting task '{task_name}': {command}")
    
    # The command already redirects stdout/stderr, but we'll capture output here
    # for Ray's logs and potential debugging.
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Finished task '{task_name}' successfully.")
    else:
        print(f"❌ ERROR in task '{task_name}': {command}\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}")
        
    return result.returncode

def generate_training_command(
    main_script_path,
    save_path,
    tasks,
    lr,
    batch_size,
    weights,
    rf_fixed,
    dropout,
    common_params,
    dataset_path
):
    """Generates the full training command string for an experiment."""
    
    # Base command
    cmd = f"python {main_script_path} "
    
    # Add tasks
    if len(tasks) > 1:
        cmd += "--mtl "
        cmd += f"--task {' '.join(tasks)} "
    else:
        cmd += f"--task {tasks[0]} "

    # Add paths and core params
    cmd += f"--pkl_dataset_path {dataset_path} "
    cmd += f"--save_path {save_path} "
    cmd += f"--epochs {common_params['epochs']} "
    cmd += f"--lr {lr} "
    cmd += f"--batch_size {batch_size} "
    cmd += f"--gpu_id 0 " # Ray sets CUDA_VISIBLE_DEVICES, so the script can always use GPU 0

    # Add model architecture params from the common dict
    for key, value in common_params.items():
        if key != 'epochs': # epochs is already added
             cmd += f"--{key} {value} "

    # Add MTL weights if any
    for task_name, weight in weights.items():
        cmd += f"--w_{task_name.split('_')[0]} {weight} " # e.g., rf_fingerprinting -> w_rf

    # Add boolean flags
    if rf_fixed:
        cmd += "--rf_fixed "
    if dropout > 0:
        cmd += f"--dropout {dropout} "
    
    cmd += "--task_adaptive_encoder "

    return cmd.strip()


def main():
    # --- Configuration ---
    MAIN_SCRIPT_PATH = "code/rep_lr/main.py"
    # Ensure PKL_FILE_PATH is set in your .env file
    dataset_path = os.path.join(os.getenv("PKL_FILE_PATH", "."), "rf_partition_dict_0.5.pkl")

    # Common model parameters for all experiments
    common_params = {
        'proj_seq_len': 256,
        'proj_hidden_dim': 512,
        'd2': 256,
        'head_hidden_dim': 256,
        'encoder_num_blocks': 2,
        'fusion_type': 'sum',
        'epochs': 300
    }

    # --- Experiment Definitions ---
    experiments = [
        {'name': 'rf', 'tasks': ['rf_fingerprinting'], 'lr': 1e-3   , 'batch_size': 512, 'weights': {}, 'dropout': 0.1},
        {'name': 'cfo', 'tasks': ['cfo_estimation'], 'lr': 1e-3, 'batch_size': 512, 'weights': {}, 'dropout': 0.0},
        {'name': 'channel', 'tasks': ['channel_estimation'], 'lr': 1e-3, 'batch_size': 256, 'weights': {}, 'dropout': 0.0},
        {'name': 'rf_cfo', 'tasks': ['rf_fingerprinting', 'cfo_estimation'], 'lr': 1e-3, 'batch_size': 512, 'weights': {'rf': 1.0, 'cfo': 1.0}, 'dropout': 0.0},
        {'name': 'rf_channel', 'tasks': ['rf_fingerprinting', 'channel_estimation'], 'lr': 1e-3, 'batch_size': 512, 'weights': {'rf': 1.0, 'channel': 1.0}, 'dropout': 0.0},
        {'name': 'cfo_channel', 'tasks': ['cfo_estimation', 'channel_estimation'], 'lr': 1e-3, 'batch_size': 512, 'weights': {'cfo': 1.0, 'channel': 1.0}, 'dropout': 0.0},
        {'name': 'rf_cfo_channel', 'tasks': ['rf_fingerprinting', 'cfo_estimation', 'channel_estimation'], 'lr': 1e-3, 'batch_size': 512, 'weights': {'rf': 1.0, 'cfo': 1.0, 'channel': 1.0}, 'dropout': 0.0},
    ]

    # --- Command Generation ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_results_dir = f"results_parallel_{timestamp}"
    os.makedirs(base_results_dir, exist_ok=True)
    print(f"Results will be saved in: {base_results_dir}")

    all_commands = []

    # Loop through both --rf_fixed and no-flag configurations
    for rf_fixed_mode in [True]: #, False]:
        mode_name = "rf_fixed" if rf_fixed_mode else "rf_variable"
        mode_dir = os.path.join(base_results_dir, mode_name)
        
        for exp in experiments:
            exp_name = exp['name']
            save_path = os.path.join(mode_dir, exp_name)
            
            # Create a dedicated log directory for clarity
            # log_dir = os.path.join(save_path, 'training_logs')
            # os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(save_path, 'train.log')
            #make the log file
            os.makedirs(save_path, exist_ok=True)
            with open(log_file, 'w') as f:
                f.write(f"==============================================\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Mode: {mode_name}\n")
                f.write(f"Tasks: {exp['tasks']}\n")
                f.write(f"LR: {exp['lr']}\n")
                f.write(f"Batch size: {exp['batch_size']}\n")
                f.write(f"Dropout: {exp['dropout']}\n")
                f.write(f"Weights: {exp['weights']}\n")
                f.write(f"RF fixed: {rf_fixed_mode}\n")
                f.write(f"Common params: {common_params}\n")
                f.write(f"Dataset path: {dataset_path}\n")
                f.write(f"concat_type: {common_params['fusion_type']}\n")
                f.write(f"Save path: {save_path}\n")
                f.write(f"==============================================\n")

            command = generate_training_command(
                main_script_path=MAIN_SCRIPT_PATH,
                save_path=save_path,
                tasks=exp['tasks'],
                lr=exp['lr'],
                batch_size=exp['batch_size'],
                weights=exp['weights'],
                rf_fixed=rf_fixed_mode,
                dropout=exp['dropout'],
                common_params=common_params,
                dataset_path=dataset_path
            )
            
            command_with_log = f"{command} > {log_file} 2>&1"
            task_name = f"{mode_name}/{exp_name}"
            all_commands.append((command_with_log, task_name))

    print(f"\nGenerated {len(all_commands)} training commands to execute.")

    # --- Ray Initialization and Execution ---
    ray_head_address = os.environ.get("RAY_HEAD_ADDRESS")
    if ray_head_address:
        print(f"Connecting to existing Ray cluster at: {ray_head_address}")
        ray.init(address=ray_head_address, runtime_env={"env_vars": safe_env_vars})
    else:
        print("Initializing Ray locally.")
        ray.init(runtime_env={"env_vars": safe_env_vars})

    print(f"Ray cluster started. Available resources: {ray.available_resources()}")

    # Submit all commands to the Ray cluster
    print("\n" + "="*50 + "\nSubmitting all experiment tasks to Ray...")
    futures = [run_command.remote(cmd, name) for cmd, name in all_commands]
    
    # Wait for all tasks to complete
    ray.get(futures)
    
    print("\n" + "="*50)
    print("🎉 All experiments are complete!")
    print(f"Find results in: {base_results_dir}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
