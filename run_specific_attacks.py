#!/usr/bin/env python3

import os
import ray
import subprocess
import time
import dotenv
import json

dotenv.load_dotenv()

# Explicitly capture environment variables from .env to pass to Ray workers
env_vars_from_dotenv = dotenv.dotenv_values()
# Ray's runtime_env expects string values, so filter out any Nones
safe_env_vars = {k: v for k, v in env_vars_from_dotenv.items() if v is not None}


def calculate_latent_dim(experiment_path):
    """
    Dynamically calculate the latent dimension based on the model architecture.
    Reads args.json from the experiment path to determine the encoder output dimension.
    """
    args_path = os.path.join(experiment_path, 'args.json')
    if not os.path.exists(args_path):
        print(f"Warning: args.json not found at {args_path}, using default latent_dim=512")
        return 512
    
    with open(args_path, 'r') as f:
        args = json.load(f)
    
    # For single-task models, the encoder output dimension is 2 * d2
    # For MTL models with concat fusion, it might be different
    d2 = args.get('d2', 128)
    is_mtl = args.get('mtl', False)
    fusion_type = args.get('fusion_type', 'sum')
    
    if is_mtl and fusion_type == 'concat':
        # For MTL with concat, the encoder input dimension is proj_seq_len * num_tasks
        # But the output dimension is still 2 * d2
        latent_dim = 2 * d2
    else:
        # For single-task models or MTL with sum fusion
        latent_dim = 2 * d2
    
    print(f"Calculated latent_dim={latent_dim} from args: d2={d2}, mtl={is_mtl}, fusion_type={fusion_type}")
    return latent_dim


@ray.remote(num_gpus=0.04) 
def run_command(command: str, task_type: str):
    """
    This worker function now only 'reserves' 0.04 of a GPU's compute.
    Ray can now schedule 25 of these to run on a single GPU simultaneously.
    """
    print(f"🚀 Starting {task_type} task on shared GPU: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Finished {task_type} task: {command}")
    else:
        print(f"❌ ERROR in {task_type} task: {command}\n   STDERR: {result.stderr}")
        
    return result.returncode


# --- Configuration ---
results_path = "/work/10608/aadharsh_aadhithya/vista/RepresentationLearning/results_20250730_220537_7"
activations_base = "/scratch/10608/aadharsh_aadhithya/data/rep_lr/activations_exps"
attack_script = "code/dra_1/robust_attack_1.py"
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Maps experiment names to activation directory names
def map_exp_to_activation_dir(exp_name):
    mapping = {
        "cfo_estimation": "cfo",
        "cfo_estimation_channel_estimation": "cfo_channel",
        "channel_estimation": "channel",
        "rf_fingerprinting": "rf",
        "rf_fingerprinting_cfo_estimation": "rf_cfo",
        "rf_fingerprinting_cfo_estimation_channel_estimation": "rf_cfo_channel",
        "rf_fingerprinting_channel_estimation": "rf_channel",
    }
    return mapping.get(exp_name, exp_name)


def find_experiment_dir(task_path):
    """Find the first subdirectory within a given path."""
    for item in os.listdir(task_path):
        item_path = os.path.join(task_path, item)
        if os.path.isdir(item_path):
            return item_path
    return None


def generate_attack_command(experiment_path, activations_path, task, output_dir, noise_type="none",
                             noise_level=0.0, fim_samples=8000, leaked_fraction=1.0, epochs=40,
                               lr=1e-3, batch_size=128, patience=30, latent_dim=None,
                               optimizer='adamw', clip_grad_norm=1.0, lambda_factor=1e-5):
    # Calculate latent_dim dynamically if not provided
    if latent_dim is None:
        latent_dim = calculate_latent_dim(experiment_path)
    
    cmd = (f"python {attack_script} --experiment_path {experiment_path} "
          f"--activations_path {activations_path} --task {task} "
          f"--output_dir {output_dir} "
          f"--noise_type {noise_type} --noise_level {noise_level} "
          f"--fim_samples {fim_samples} --leaked_fraction {leaked_fraction} "
          f"--epochs {epochs} --lr {lr} --batch_size {batch_size} "
          f"--patience {patience} --latent_dim {latent_dim} "
          f"--optimizer {optimizer} --clip_grad_norm {clip_grad_norm} "
          f"--use_lr_scheduler")

    if noise_type == 'nonisotropic':
        cmd += f" --lambda_factor {lambda_factor}"
           
    return cmd


def main():
    attack_commands = []
    
    # Define specific attack configurations
    attack_configs = [
        # cfo_estimation+channel_estimation leaked fraction 1 isotropic level 15
        ("cfo_estimation_channel_estimation", "cfo", "isotropic", 15, 1.0, None),
        ("cfo_estimation_channel_estimation", "channel", "isotropic", 15, 1.0, None),
        
        # rf fingerprinting leaked fraction 1 nonisotropic level 15 lambda_fraction 0.01
        ("rf_fingerprinting", "rf", "nonisotropic", 15, 1.0, 0.01),
        
        # rf fingerprinting leaked fraction 1 nonisotropic level 10 lambda_fraction 0.01
        ("rf_fingerprinting", "rf", "nonisotropic", 10, 1.0, 0.01),
        
        # rf fingerprinting leaked fraction 1 isotropic level 20
        ("rf_fingerprinting", "rf", "isotropic", 20, 1.0, None),
        
        # rf fingerprinting leaked fraction 1 isotropic level 5
        ("rf_fingerprinting", "rf", "isotropic", 5, 1.0, None),
        
        # rf+channel leaked fraction 1 nonisotropic 10 lambda 0.01
        ("rf_fingerprinting_channel_estimation", "rf", "nonisotropic", 10, 1.0, 0.01),
        ("rf_fingerprinting_channel_estimation", "channel", "nonisotropic", 10, 1.0, 0.01),
    ]
    
    for exp_name, task, noise_type, noise_level, leaked_fraction, lambda_factor in attack_configs:
        print(f"\n--- Processing: {exp_name}, task={task}, noise={noise_type}, level={noise_level}, lambda={lambda_factor} ---")
        
        # Find experiment directory
        dir_name = map_exp_to_activation_dir(exp_name)
        task_path = os.path.join(results_path, dir_name)
        if not os.path.isdir(task_path):
            print(f"ERROR: Task path does not exist: {task_path}")
            continue
            
        full_experiment_path = find_experiment_dir(task_path)
        if not full_experiment_path:
            print(f"ERROR: Could not find experiment directory in {task_path}")
            continue
        
        # Get activation path
        results_name = os.path.basename(results_path)
        activation_dir_name = map_exp_to_activation_dir(exp_name)
        activations_path = os.path.join(activations_base, results_name, activation_dir_name)
        
        if not os.path.isdir(activations_path):
            print(f"ERROR: Activations path does not exist: {activations_path}")
            continue
        
        print(f"Using activations from: {activations_path}")
        
        # Construct output directory
        frac_str = str(leaked_fraction).replace('.', '_')
        level_str = str(float(noise_level)).replace('.', '_')
        base_attack_dir = os.path.join(full_experiment_path, 'attack_results_robust', f'attack_{timestamp}')
        results_dir = os.path.join(base_attack_dir, task, noise_type, f'leaked_frac_{frac_str}', f'noise_level_{level_str}')
        
        if noise_type == 'nonisotropic' and lambda_factor is not None:
            lambda_str = f"lambda_{str(lambda_factor).replace('.', '_')}"
            results_dir = os.path.join(results_dir, lambda_str)
            
        # Create output directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate attack command
        attack_command = generate_attack_command(
            experiment_path=full_experiment_path,
            activations_path=activations_path,
            task=task,
            output_dir=results_dir,
            noise_type=noise_type,
            noise_level=noise_level,
            leaked_fraction=leaked_fraction,
            optimizer='adamw',
            clip_grad_norm=1.0,
            lambda_factor=lambda_factor if lambda_factor else 1e-5
        )
        
        # Add log file redirection
        log_file_path = os.path.join(results_dir, 'attack.log')
        attack_command_with_log = f"{attack_command} > {log_file_path} 2>&1"
        attack_commands.append(attack_command_with_log)
        
        print(f"Generated command for {exp_name}/{task}")
        print(f"Output dir: {results_dir}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Generated {len(attack_commands)} attack commands")
    
    if not attack_commands:
        print("No commands to run. Exiting.")
        return
    
    # Print all commands for verification
    print("\nCommands to run:")
    for i, cmd in enumerate(attack_commands, 1):
        print(f"{i}. {cmd}")
    
    # Initialize Ray and run commands
    ray_head_address = os.environ.get("RAY_HEAD_ADDRESS")
    if ray_head_address:
        print(f"\nConnecting to existing Ray cluster at: {ray_head_address}")
        ray.init(
            address=ray_head_address,
            runtime_env={
                "conda": "vllm",
                "env_vars": safe_env_vars,
            }
        )
    else:
        print("RAY_HEAD_ADDRESS not found, initializing Ray locally.")
        ray.init(
            runtime_env={
                "conda": "vllm",
                "env_vars": safe_env_vars,
            }
        )

    print(f"\nRay cluster started. Available resources: {ray.available_resources()}")

    # Submit all attack tasks
    print("\n" + "="*50 + "\nSubmitting attack tasks.")
    attack_futures = [run_command.remote(cmd, "attack") for cmd in attack_commands]
    ray.get(attack_futures)
    print("\n🎉 All specific attacks are complete!")


if __name__ == "__main__":
    main()