import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.models import load_trained_model
from dra_1.py_datasets import ActivationDataset

def inject_noise(z, strength):
    """
    Injects white Gaussian noise into a tensor.
    Args:
        z (torch.Tensor): The input tensor (activation).
        strength (float): The variance of the Gaussian noise.
    Returns:
        torch.Tensor: The tensor with added noise.
    """
    if strength <= 0:
        return z
    noise = torch.randn_like(z) * (strength**0.5)
    return z + noise

def reconstruct_batch(trained_model, other_task_inputs, target_activation, initial_x0, 
                                 is_mtl, train_args, attack_args):
    """
    Performs the optimization-based reconstruction attack for a batch of samples.
    """
    x_rec = initial_x0.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x_rec], lr=attack_args.lr)
    criterion = nn.MSELoss()

    TASK_NAME_MAPPING = {
        'rf': 'rf_fingerprinting',
        'cfo': 'cfo_estimation',
        'channel': 'channel_estimation'
    }

    full_target_task_name = TASK_NAME_MAPPING.get(attack_args.task)
    if not full_target_task_name:
        raise ValueError(f"Invalid task name: {attack_args.task}")

    if is_mtl:
        # Calculate projections from other tasks, which are fixed during optimization
        projected_others = 0
        for task_short_name, other_x in other_task_inputs.items():
            full_task_name = TASK_NAME_MAPPING[task_short_name]
            proj_other = trained_model['projections'][full_task_name](other_x)
            projected_others += proj_other
        
        projection_layer = trained_model['projections'][full_target_task_name]
    else:
        projection_layer = trained_model['projection']
    
    encoder_layer = trained_model['encoder']

    for _ in range(attack_args.optim_steps):
        optimizer.zero_grad()
        
        projected_target = projection_layer(x_rec)
        
        # In MTL, the encoder input is the sum of all projections
        if is_mtl:
            projected_sum = projected_target + projected_others
            z_hat = encoder_layer(projected_sum)
        else: # In single-task, it's just the one projection
            z_hat = encoder_layer(projected_target)
        
        loss = criterion(z_hat, target_activation)
        loss.backward()
        optimizer.step()

    return x_rec.detach()

def visualize_reconstruction(original_samples, reconstructed_samples, task_name):
    """
    Generates and displays plots comparing original and reconstructed signals.
    """
    if not original_samples:
        print("No samples to visualize.")
        return

    num_samples_to_plot = len(original_samples)
    fig, axes = plt.subplots(num_samples_to_plot, 3, figsize=(18, 3 * num_samples_to_plot), squeeze=False)
    fig.suptitle(f"Reconstruction Results for Task: '{task_name}'", fontsize=16)

    for i in range(num_samples_to_plot):
        original_np = original_samples[i].squeeze(0).numpy()
        recon_np = reconstructed_samples[i].squeeze(0).numpy()

        # Plot Real Part
        axes[i, 0].plot(original_np[0][:100], label='Original')
        axes[i, 0].plot(recon_np[0][:100], label='Reconstructed', linestyle='--')
        axes[i, 0].set_ylabel(f"Sample {i+1}")
        axes[i, 0].set_title("Real Part")
        axes[i, 0].legend()

        # Plot Imaginary Part
        axes[i, 1].plot(original_np[1][:100], label='Original')
        axes[i, 1].plot(recon_np[1][:100], label='Reconstructed', linestyle='--')
        axes[i, 1].set_title("Imaginary Part")
        axes[i, 1].legend()

        # Plot Magnitude
        original_mag = np.sqrt(original_np[0]**2 + original_np[1]**2)
        recon_mag = np.sqrt(recon_np[0]**2 + recon_np[1]**2)
        axes[i, 2].plot(original_mag[:100], label='Original')
        axes[i, 2].plot(recon_mag[:100], label='Reconstructed', linestyle='--')
        axes[i, 2].set_title("Magnitude")
        axes[i, 2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"reconstruction_results_{task_name}.png")

def check_utility(z_clean, z_noisy, trained_model, is_mtl, train_args, labels, device):
    """
    Checks task utility (performance) on clean and noisy activations by computing the loss
    for each task.
    
    Args:
        z_clean (torch.Tensor): The clean activation tensor (before noise).
        z_noisy (torch.Tensor): The noisy activation tensor (after noise).
        trained_model (dict): The dictionary containing model components.
        is_mtl (bool): Flag indicating if the model is multi-task.
        train_args (Namespace): Training arguments, used to identify tasks.
        labels (dict): A dictionary of ground truth labels for each task.
        device (torch.device): The device to run computations on.

    Returns:
        dict: A dictionary with utility results (losses) for each task.
    """
    utility_results = {}
    
    TASK_NAME_MAPPING = {
        'rf': 'rf_fingerprinting',
        'cfo': 'cfo_estimation',
        'channel': 'channel_estimation'
    }

    # Handle DataParallel wrapper
    model_to_check = trained_model.get('module', trained_model)

    if is_mtl:
        tasks_to_check = model_to_check.get('task_heads', {})
    else:
        full_task_name = TASK_NAME_MAPPING.get(train_args.task)
        if full_task_name and 'task_head' in model_to_check:
            tasks_to_check = {full_task_name: model_to_check['task_head']}
        else:
            tasks_to_check = {}

    for task_name, head in tasks_to_check.items():
        short_task_name = [k for k, v in TASK_NAME_MAPPING.items() if v == task_name]
        if not short_task_name or short_task_name[0] not in labels:
            continue
        short_task_name = short_task_name[0]

        y_true = labels[short_task_name].to(device)

        if task_name == 'rf_fingerprinting':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            y_true = y_true.squeeze(1).long()
        else:
            loss_fn = nn.MSELoss(reduction='none')
            if y_true.dim() > 2:
                y_true = y_true.squeeze(1)

        with torch.no_grad():
            # Utility with clean activations
            y_pred_clean = head(z_clean)
            loss_clean = loss_fn(y_pred_clean, y_true)

            # Utility with noisy activations
            y_pred_noisy = head(z_noisy)
            loss_noisy = loss_fn(y_pred_noisy, y_true)
        
        utility_results[task_name] = {
            'clean_loss': loss_clean.cpu(),
            'noisy_loss': loss_noisy.cpu()
        }
        
    return utility_results

def run_attack(cli_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained (and frozen) victim model
    trained_model, train_args, is_mtl = load_trained_model(cli_args.model_checkpoint_path, device)

    # Load the dataset containing activations
    with open(cli_args.partition_file, 'rb') as f:
        partitions = pickle.load(f)
    
    test_files = partitions.get('test', [])
    if not test_files:
        raise ValueError("Could not find a 'test' key in the partition file.")

    # Use a subset for faster evaluation if specified
    if cli_args.test_fraction < 1.0:
        num_samples = int(len(test_files) * cli_args.test_fraction)
        test_files = test_files[:num_samples]

    test_dataset = ActivationDataset(
        activation_dir=cli_args.activation_dir,
        file_list=test_files,
        test_mode=True
    )
    test_loader = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False, num_workers=4)

    total_recon_loss = 0
    total_utility_losses = {'clean': defaultdict(float), 'noisy': defaultdict(float)}
    criterion = nn.MSELoss(reduction='none')

    reconstruction_results = []
    original_samples = []
    reconstructed_samples = []
    
    for rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation_batch, filename_batch in tqdm(test_loader, desc="Running Reconstruction Attack"):
        z_clean = activation_batch.squeeze(1).to(device)
        
        # Inject noise if specified
        z_noisy = inject_noise(z_clean, cli_args.noise_strength)
        
        # The attack proceeds with the noisy activations
        activation_batch = z_noisy

        # Check utility before and after adding noise
        labels = {
            'rf': rf_y,
            'cfo': cfo_y,
            'channel': channel_y,
        }
        utility_results = check_utility(z_clean, z_noisy, trained_model, is_mtl, train_args, labels, device)

        print(utility_results)

        for task_name, results in utility_results.items():
            total_utility_losses['clean'][task_name] += results['clean_loss'].sum().item()
            total_utility_losses['noisy'][task_name] += results['noisy_loss'].sum().item()
        
        all_true_inputs = {
            'rf': rf_x.squeeze(1).to(device),
            'cfo': cfo_x.squeeze(1).to(device),
            'channel': channel_x.squeeze(1).to(device)
        }
        
        # The data we are trying to reconstruct
        true_x_batch = all_true_inputs.pop(cli_args.task)
        # The data from other tasks, needed for the MTL forward pass
        other_task_inputs = all_true_inputs if is_mtl else {}

        # Initialize a random tensor as the starting point for optimization
        x0_batch = torch.randn_like(true_x_batch)

        # Reconstruct the input
        reconstructed_x_batch = reconstruct_batch(
            trained_model, other_task_inputs, activation_batch, x0_batch, is_mtl, train_args, cli_args
        )

        # Calculate the final reconstruction loss against the true data
        recon_loss_per_sample = criterion(reconstructed_x_batch.flatten(1), true_x_batch.flatten(1)).mean(dim=1)
        total_recon_loss += recon_loss_per_sample.sum().item()
        
        # Store a few samples for later visualization
        if not original_samples:
            num_to_visualize = min(5, true_x_batch.size(0))
            original_samples.extend(list(torch.split(true_x_batch.cpu(), 1))[:num_to_visualize])
            reconstructed_samples.extend(list(torch.split(reconstructed_x_batch.cpu(), 1))[:num_to_visualize])
        
        for i in range(len(filename_batch)):
            reconstruction_results.append({
                'filename': filename_batch[i],
                'reconstruction_mse': recon_loss_per_sample[i].item(),
            })
        #print batch MSE loss
        print(f"Batch MSE loss: {recon_loss_per_sample.mean().item()}")

    avg_recon_loss = total_recon_loss / len(test_dataset)
    print(f"\nAverage Data Reconstruction MSE: {avg_recon_loss:.6f}")
    
    # Print utility results
    print("\n--- Task Utility ---")
    num_samples = len(test_dataset)
    for task_name in total_utility_losses['clean']:
        avg_clean_loss = total_utility_losses['clean'][task_name] / num_samples
        avg_noisy_loss = total_utility_losses['noisy'][task_name] / num_samples
        print(f"Task: {task_name}")
        print(f"  - Avg Loss (Clean Activations): {avg_clean_loss:.6f}")
        print(f"  - Avg Loss (Noisy Activations): {avg_noisy_loss:.6f}")

    # Visualize the reconstruction results for the saved samples
    visualize_reconstruction(original_samples, reconstructed_samples, cli_args.task)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimization-based Data Reconstruction Attack.")
    
    # Required paths
    parser.add_argument('--model_checkpoint_path', type=str, required=True, help="Path to the victim model's .pt checkpoint.")
    parser.add_argument('--activation_dir', type=str, required=True, help="Directory containing pre-computed activation files.")
    parser.add_argument('--partition_file', type=str, required=True, help="Path to the data partition file.")
    
    # Attack parameters
    parser.add_argument('--task', type=str, required=True, choices=['rf', 'cfo', 'channel'], help="The target task to reconstruct data from.")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for the input optimization.")
    parser.add_argument('--optim_steps', type=int, default=2000, help="Number of optimization steps per sample.")
    parser.add_argument('--noise_strength', type=float, default=0.2, help="Variance of Gaussian noise to add to activations before attack.")

    # Evaluation options
    parser.add_argument('--test_fraction', type=float, default=0.1, help="Fraction of the test set to use for the attack.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for the attack.")
    
    cli_args = parser.parse_args()
    run_attack(cli_args) 