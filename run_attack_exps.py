import os
import ray
import subprocess
import time
import dotenv

dotenv.load_dotenv()

# Explicitly capture environment variables from .env to pass to Ray workers
env_vars_from_dotenv = dotenv.dotenv_values()
# Ray's runtime_env expects string values, so filter out any Nones
safe_env_vars = {k: v for k, v in env_vars_from_dotenv.items() if v is not None}


@ray.remote(num_gpus=0.04) 
def run_command(command: str, task_type: str):
    """
    This worker function now only 'reserves' 0.1 of a GPU's compute.
    Ray can now schedule 10 of these to run on a single GPU simultaneously.
    """
    print(f"🚀 Starting {task_type} task on shared GPU: {command}")
    
    # The rest of the function is the same
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Finished {task_type} task: {command}")
    else:
        print(f"❌ ERROR in {task_type} task: {command}\n   STDERR: {result.stderr}")
        
    return result.returncode

remote_activation = True #STrue


results_path =  "/work/10608/aadharsh_aadhithya/vista/RepresentationLearning/results_20250725_111527/run_1"
#results_path =  "/home/hofmann/Documents/projects/RepresentationLearning/results_20250720_172807"
activations_base =  os.getenv("ACTIVATIONS_BASE")# "/home/hofmann/Documents/projects/RepresentationLearning/results_20250720_172807"
extract_activation_script = os.getenv("EXTRACT_ACTIVATION_SCRIPT")# "/home/hofmann/Documents/projects/RepresentationLearning/code/dra_1/extract_activations.py"
attack_script = os.getenv("ATTACK_SCRIPT")# "/home/hofmann/Documents/projects/RepresentationLearning/code/dra_1/robust_attack.py"
is_uv=os.getenv("IS_UV")
ray_tmp_dir = os.getenv("RAY_TMP_DIR")# "/home/hofmann/Documents/ray_temp"

if not os.path.exists(ray_tmp_dir):
    os.makedirs(ray_tmp_dir)


tasks = os.listdir(results_path)


#  # Paths and identifiers
#     parser.add_argument('--experiment_path', type=str, required=True, help="Path to the experiment directory.")
#     parser.add_argument('--activations_path', type=str, default=None, help="Path to activations directory. If not provided, will use <experiment_path>/activations.")
    
#     # Task and Noise
#     parser.add_argument('--task', type=str, required=True, choices=['rf', 'cfo', 'channel'], help="Task to attack.")
#     parser.add_argument('--noise_type', type=str, default='none', choices=['isotropic', 'nonisotropic', 'none'], help="Type of noise to inject.")
#     parser.add_argument('--noise_level', type=float, default=0.0, help="Total variance of the noise.")
#     parser.add_argument('--fim_samples', type=int, default=1000, help="Number of samples for FIM calculation.")
#     parser.add_argument('--leaked_fraction', type=float, default=1.0, help="Fraction of the training dataset that is considered 'leaked' and used for the attack.")

#     # Training parameters
#     parser.add_argument('--epochs', type=int, default=30, help="Max number of training epochs.")
#     parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the decoder.")
#     parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
#     parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping.")

#     # Model parameters
#     parser.add_argument('--latent_dim', type=int, default=512, help="Dimension of the latent space.")
    



def generate_attack_command(experiment_path, activations_path, task, output_dir, noise_type="none",
                             noise_level=0.0, fim_samples=8000, leaked_fraction=1.0, epochs=40,
                               lr=1e-3, batch_size=128, patience=30, latent_dim=512,
                               optimizer='adamw', clip_grad_norm=1.0, lambda_factor=1e-5):
    cmd= (f"python {attack_script} --experiment_path {experiment_path} "
          f"--activations_path {activations_path} --task {task} "
          f"--output_dir {output_dir} "
          f"--noise_type {noise_type} --noise_level {noise_level} "
          f"--fim_samples {fim_samples} --leaked_fraction {leaked_fraction} "
          f"--epochs {epochs} --lr {lr} --batch_size {batch_size} "
          f"--patience {patience} --latent_dim {latent_dim} "
          f"--optimizer {optimizer} --clip_grad_norm {clip_grad_norm} "
          f"--lambda_factor {lambda_factor}")

    # if is_uv:
    #     cmd = "uv run " + cmd
    return cmd


def generate_activation_command(model_path, output_dir, gpu_id=0, data_fraction=1.0, overwrite=False):
    cmd = f"python {extract_activation_script} --model_path {model_path} --output_dir {output_dir} --gpu_id {gpu_id} --data_fraction {data_fraction}"

    # if is_uv:
    #     cmd = "uv run " + cmd
    return cmd

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    activation_commands = [] 
    attack_commands = []
    experiment_tasks = []

    for task in tasks:
        possible_tasks = ["rf","cfo", "channel"] #removing rf for now, as it is not working, will save time
        maybe_tasks = task.split("_")

        if maybe_tasks[0] in possible_tasks:
            experiment_tasks.append(maybe_tasks)



    for task in experiment_tasks:

        #skip loop if len of task grater than 1
        

        activations_path = os.path.join(activations_base, )
        task_path = os.path.join(results_path, "_".join(task))
        task_base_path = os.listdir(task_path)[0]
        
        full_task_base_path = os.path.join(task_path, task_base_path)

        if remote_activation:
            results_name = results_path.split("/")[-1]
            activations_path = os.path.join(os.path.join(activations_base, results_name),"_".join(task))
        else:
            activations_path = os.path.join(full_task_base_path, "activations")

        log_dir = os.path.join(full_task_base_path, "attack_logs")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(activations_path):
            print(f"Activations path {activations_path} does not exist")
            os.makedirs(activations_path)

        else:
            print(f"Activations path {activations_path} exists")

        
        

        #see if activation path is empty
        if not os.listdir(activations_path):
            print(f"Activations path {activations_path} is empty")
            

            act_command = generate_activation_command(full_task_base_path, activations_path)
            #add log file to act_command
            act_command = act_command + f" > {log_dir}/activations.log 2>&1"
            activation_commands.append(act_command)

            
            #have thing to run command here

        print(f"Generating attack commands for {task}")
        #task-> [t1, t2, t3]
        noise_types = ["none","isotropic",'nonisotropic']
        
        leaked_fractions = [0.1,0.5,1.0]
        lambda_factors = [1e-1,1e-2, 1e-3,1e-4] # Add your desired sweep values here


        for noise_type in noise_types:
            if noise_type == "none":
                noise_levels = [0]
            else:
                noise_levels = [5,10,15,20]
            
            for noise_level in noise_levels:
                for leaked_fraction in leaked_fractions:
                    for t in task:
                        #do not attack rf for now
                        if t == "rf":
                            continue
                        
                        # Construct the results directory path
                        frac_str = str(leaked_fraction).replace('.', '_')
                        level_str = str(float(noise_level)).replace('.', '_')

                        # Create a timestamped base directory for this run's attack results
                        base_attack_dir = os.path.join(full_task_base_path, 'attack_results_robust', f'attack_{timestamp}')

                        # Define the final results directory for this specific configuration
                        if noise_type == 'nonisotropic':
                            for lambda_factor in lambda_factors:
                                lambda_str = f"lambda_{str(lambda_factor).replace('.', '_')}"
                                results_dir = os.path.join(base_attack_dir, t, noise_type, f'leaked_frac_{frac_str}', f'noise_level_{level_str}', lambda_str)
                                
                                attack_command = generate_attack_command(experiment_path=full_task_base_path,
                                                                        activations_path=activations_path,
                                                                        task=t,
                                                                        output_dir=results_dir,
                                                                        noise_type=noise_type,
                                                                        noise_level=noise_level,
                                                                        leaked_fraction=leaked_fraction,
                                                                        optimizer='adamw',
                                                                        clip_grad_norm=1.0,
                                                                        lambda_factor=lambda_factor)
                                
                                # Ensure the directory exists
                                os.makedirs(results_dir, exist_ok=True)
                                
                                # Define the log file path and redirect output
                                log_file_path = os.path.join(results_dir, 'attack.log')
                                print(f"Log file path: {log_file_path}")
                                attack_command = attack_command + f" > {log_file_path} 2>&1"
                                attack_commands.append(attack_command)
                        else:
                            results_dir = os.path.join(base_attack_dir, t, noise_type, f'leaked_frac_{frac_str}', f'noise_level_{level_str}')
                            
                            attack_command = generate_attack_command(experiment_path=full_task_base_path,
                                                                    activations_path=activations_path,
                                                                    task=t,
                                                                    output_dir=results_dir,
                                                                    noise_type=noise_type,
                                                                    noise_level=noise_level,
                                                                    leaked_fraction=leaked_fraction,
                                                                    optimizer='adamw',
                                                                    clip_grad_norm=1.0)
                            
                            # Ensure the directory exists
                            os.makedirs(results_dir, exist_ok=True)
                            
                            # Define the log file path and redirect output
                            log_file_path = os.path.join(results_dir, 'attack.log')
                            print(f"Log file path: {log_file_path}")
                            attack_command = attack_command + f" > {log_file_path} 2>&1"
                            attack_commands.append(attack_command)


    print(f"Generated {len(activation_commands)} activation commands and {len(attack_commands)} attack commands")

        



    ray.init(
        _temp_dir=ray_tmp_dir,
        runtime_env={
            "conda": "vllm",
            # "py_executable": "/home/hofmann/Documents/projects/RepresentationLearning/.venv/bin/python3",
            "env_vars": safe_env_vars,
        }
    )
    print(f"Ray cluster started. Available resources: {ray.available_resources()}")

    # --- PHASE 1: Run Activations ---
    print("\n" + "="*50 + "\nPHASE 1: Submitting activation tasks.")
    # Here, Ray will run all 3 activation tasks concurrently on the single GPU
    # since 3 * 0.1 = 0.3, which is < 1.0.
    if len(activation_commands) > 0:
        activation_futures = [run_command.remote(cmd, "activation") for cmd in activation_commands]
        ray.get(activation_futures)
        print("\n✅ All activation tasks finished.")

    # --- PHASE 2: Run Attacks ---
    print("\n" + "="*50 + "\nPHASE 2: Submitting attack tasks.")
    # Ray will run 10 attack tasks at a time on the GPU. As soon as one
    # finishes, the next one from the queue of 50 will start.
    if len(attack_commands) > 0:
        attack_futures = [run_command.remote(cmd, "attack") for cmd in attack_commands]
        ray.get(attack_futures)
        print("\n🎉 All experiments are complete!")


if __name__ == "__main__":
    main()
    

        


