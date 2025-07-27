import os
import sys
import json
import argparse
import torch
import numpy as np
import pickle
import random
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
import torch.nn as nn

# Add project root to path to allow importing from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.py_datasets import ActivationDataset
from dra_1.inject_noise import inject_isotropic_noise, inject_nonisotropic_noise
from dra_1.model_loader import load_model_for_utility_check
from rep_lr.models import ChannelEstimationHead

def get_empirical_fim(head, data_loader, device, latent_dim):
    """Calculates the empirical Fisher Information Matrix for channel estimation head."""
    fim = torch.zeros((latent_dim, latent_dim), device=device)
    num_samples = 0
    head.to(device).eval()

    # Use the same complex MSE loss as in training for consistency
    mse_loss = nn.MSELoss()
    def complex_mse_loss(pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        return mse_loss(pred_flat, target_flat)
    criterion = complex_mse_loss

    for batch in tqdm(data_loader, desc="Calculating FIM"):
        # Unpack based on what ActivationDataset yields in train mode
        _, rf_labels, _, cfo_labels, _, ch_labels, activations, _ = batch
        
        activations = activations.squeeze(1).to(device)
        activations = activations.view(activations.size(0), -1)
        activations.requires_grad = True # Ensure we can get gradients

        head.zero_grad()
        
        # Channel estimation target
        y_target = ch_labels.to(device)
        # Fix channel label shape: [batch, 1, 2, 52] -> [batch, 2, 52]
        if y_target.dim() == 4 and y_target.size(1) == 1:
            y_target = y_target.squeeze(1)

        y_hat = head(activations)
        loss = criterion(y_hat, y_target)
        loss.backward()
        
        if activations.grad is not None:
            J = activations.grad
            fim += J.T @ J
            num_samples += activations.size(0)

    if num_samples == 0:
        return fim
    return fim / num_samples

def evaluate_channel_utility(head, test_dl, device, output_dir, noise_level, L=None, V=None, noise_type='isotropic'):
    """Evaluate channel estimation utility with noise injection."""
    head.to(device).eval()
    all_y_true, all_y_pred, evaluation_results = [], [], []
    
    # Define the same complex MSE loss as used in training
    mse_loss = nn.MSELoss()
    def complex_mse_loss(pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        return mse_loss(pred_flat, target_flat)
    
    total_training_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for _, _, _, _, _, labels, activations, file_paths in tqdm(test_dl, desc="Evaluating Channel Utility"):
            activations = activations.squeeze(1).to(device)
            labels = labels.to(device)
            
            activations = activations.view(activations.size(0), -1)

            if noise_type == 'isotropic':
                noisy_activations = inject_isotropic_noise(activations, noise_level)
            elif noise_type == 'nonisotropic':
                noisy_activations = inject_nonisotropic_noise(activations, noise_level, L, V)
            else:
                noisy_activations = activations
            
            # Reshape back to (batch, 2, dim) for the head
            noisy_activations = noisy_activations.view(noisy_activations.size(0), 2, -1)

            outputs = head(noisy_activations)
            
            # Compute training loss for this batch
            batch_loss = complex_mse_loss(outputs, labels)
            total_training_loss += batch_loss.item() * labels.size(0)
            num_samples += labels.size(0)
            
            for i in range(labels.size(0)):
                true_val = labels[i].cpu().numpy()
                pred_val = outputs[i].cpu().numpy()
                all_y_true.append(true_val)
                all_y_pred.append(pred_val)
                evaluation_results.append({
                    'filename': os.path.basename(file_paths[i]),
                    'y_true': true_val.tolist(), 'y_pred': pred_val.tolist(),
                })

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    # Reshape the arrays to treat the two channels as real and imaginary parts
    # The expected shape for y_true and y_pred is (num_samples, 2, 52)
    # If it comes in as (num_samples, 1, 104), we reshape it.
    if y_true.shape[1] == 1 and y_true.shape[2] == 104:
        y_true = y_true.reshape(y_true.shape[0], 2, 52)
    if y_pred.shape[1] == 1 and y_pred.shape[2] == 104:
        y_pred = y_pred.reshape(y_pred.shape[0], 2, 52)

    if len(y_true.shape) == 4 and y_true.shape[1] == 1:
        y_true = np.squeeze(y_true, axis=1)

    y_true_complex = y_true[:, 0, :] + 1j * y_true[:, 1, :]
    y_pred_complex = y_pred[:, 0, :] + 1j * y_pred[:, 1, :]

    nmse = np.mean(np.abs(y_true_complex - y_pred_complex)**2) / np.mean(np.abs(y_true_complex)**2)
    nmse_db = 10 * np.log10(nmse)
    
    y_true_real = y_true[:, 0, :]
    y_pred_real = y_pred[:, 0, :]
    y_true_imag = y_true[:, 1, :]
    y_pred_imag = y_pred[:, 1, :]

    mae_real = mean_absolute_error(y_true_real, y_pred_real)
    mse_real = mean_squared_error(y_true_real, y_pred_real)
    r2_real = r2_score(y_true_real, y_pred_real)

    mae_imag = mean_absolute_error(y_true_imag, y_pred_imag)
    mse_imag = mean_squared_error(y_true_imag, y_pred_imag)
    r2_imag = r2_score(y_true_imag, y_pred_imag)
    
    # Compute average training loss
    avg_training_loss = total_training_loss / num_samples if num_samples > 0 else 0.0
    
    metrics = {
        'training_loss': float(avg_training_loss),  # Same loss as used in training
        'nmse': float(nmse),
        'nmse_db': float(nmse_db),
        'real_part': {
            'mae': float(mae_real),
            'mse': float(mse_real),
            'r2_score': float(r2_real)
        },
        'imaginary_part': {
            'mae': float(mae_imag),
            'mse': float(mse_imag),
            'r2_score': float(r2_imag)
        }
    }
    
    print("\nChannel Estimation Metrics:")
    print(f"Training Loss: {avg_training_loss:.4f}")  # Same loss as used in training
    print(f"NMSE: {nmse:.4f}")
    print(f"NMSE (dB): {nmse_db:.4f}")
    print("\nReal Part Metrics:")
    for key, value in metrics['real_part'].items():
        print(f"{key.upper()}: {value:.4f}")
    print("\nImaginary Part Metrics:")
    for key, value in metrics['imaginary_part'].items():
        print(f"{key.upper()}: {value:.4f}")

    with open(os.path.join(output_dir, 'channel_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(output_dir, 'channel_predictions.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Channel metrics and predictions saved in {output_dir}")

def run_channel_evaluations(cli_args, head, test_dl, train_files, activation_dir):
    """Run channel estimation evaluations for all noise levels and types."""
    device = torch.device(f'cuda:{cli_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    noise_levels = np.linspace(0, 10, 11)  # 0, 1, 2, ..., 10
    noise_types = ['isotropic', 'nonisotropic']

    # --- Pre-calculate FIM for non-isotropic noise ---
    L, V = None, None
    if 'nonisotropic' in noise_types:
        print("\n===== Pre-calculating FIM for Anisotropic Noise =====")
        if not train_files:
            raise ValueError("Training files are required for FIM calculation but not found in partition.")
        
        # Use a subset of training data for FIM
        random.shuffle(train_files)
        fim_files = train_files[:cli_args.fim_samples]
        fim_dataset = ActivationDataset(activation_dir=activation_dir, file_list=fim_files, test_mode=False)
        fim_dl = DataLoader(fim_dataset, batch_size=cli_args.batch_size, shuffle=True)
        
        latent_dim = head.input_dim
        print(f"Calculating FIM for channel estimation head (latent_dim: {latent_dim})")
        fim = get_empirical_fim(head, fim_dl, device, latent_dim)
        
        print("Performing eigendecomposition of FIM...")
        L_e, V = torch.linalg.eigh(fim)
        L = torch.relu(L_e)  # Ensure non-negative eigenvalues
        print("FIM calculation and decomposition complete.")

    # --- Run evaluation loop ---
    results_r2 = {'isotropic': [], 'nonisotropic': []}
    results_mse = {'isotropic': [], 'nonisotropic': []}
    
    for noise_type in noise_types:
        for noise_level in tqdm(noise_levels, desc=f"Evaluating {noise_type} noise"):
            
            # Skip FIM-based evaluation if data is not available
            if noise_type == 'nonisotropic' and L is None:
                continue

            # Determine the final output directory path
            base_output_dir = cli_args.output_dir
            if base_output_dir is None:
                base_output_dir = os.path.join(cli_args.experiment_path, 'channel_debug_exps')
            
            noise_level_str = f"level_{str(noise_level).replace('.', '_')}"
            output_dir = os.path.join(base_output_dir, noise_type, noise_level_str)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n===== Evaluating Channel Estimation (Noise: {noise_type}, Level: {noise_level}) =====")
            print(f"Saving results to: {output_dir}")
            
            evaluate_channel_utility(head, test_dl, device, output_dir, noise_level, L, V, noise_type)
            
            # Collect results for plotting
            metrics_file = os.path.join(output_dir, 'channel_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    # Average the R² and MSE of real and imaginary parts
                    r2_metric = (metrics['real_part']['r2_score'] + metrics['imaginary_part']['r2_score']) / 2.0
                    mse_metric = (metrics['real_part']['mse'] + metrics['imaginary_part']['mse']) / 2.0
                    results_r2[noise_type].append(r2_metric)
                    results_mse[noise_type].append(mse_metric)
    
    print("\nChannel estimation evaluations complete.")
    return base_output_dir, results_r2, results_mse

def plot_channel_results(base_output_dir, results_r2, results_mse):
    """Plot channel estimation results."""
    print("\n--- Generating channel estimation plots ---")
    noise_levels = np.linspace(0, 10, 11)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.get_cmap('viridis', 4)

    # Plot R² scores
    plt.figure(figsize=(12, 7))
    plt.plot(noise_levels, results_r2['isotropic'], 'o-', color=colors(0.3), linewidth=2, markersize=8, label='Isotropic Noise')
    plt.plot(noise_levels, results_r2['nonisotropic'], 's--', color=colors(0.7), linewidth=2, markersize=8, label='Non-isotropic Noise')

    plt.title('Channel Estimation R² Score vs Noise Level', fontsize=16, fontweight='bold')
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plot_path = os.path.join(base_output_dir, 'channel_estimation_r2_vs_noise.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved R² plot: {plot_path}")
    plt.close()

    # Plot MSE scores
    plt.figure(figsize=(12, 7))
    plt.plot(noise_levels, results_mse['isotropic'], 'o-', color=colors(0.3), linewidth=2, markersize=8, label='Isotropic Noise')
    plt.plot(noise_levels, results_mse['nonisotropic'], 's--', color=colors(0.7), linewidth=2, markersize=8, label='Non-isotropic Noise')

    plt.title('Channel Estimation MSE vs Noise Level', fontsize=16, fontweight='bold')
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plot_path = os.path.join(base_output_dir, 'channel_estimation_mse_vs_noise.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved MSE plot: {plot_path}")
    plt.close()

    # Save aggregated results to JSON
    aggregated_results = {
        'noise_levels': noise_levels.tolist(),
        'r2_scores': results_r2,
        'mse_scores': results_mse
    }
    
    results_json_path = os.path.join(base_output_dir, 'channel_estimation_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(aggregated_results, f, indent=4)
    print(f"Saved aggregated results to: {results_json_path}")

def main():
    parser = argparse.ArgumentParser(description='Debug channel estimation utility evaluations across noise levels.')
    parser.add_argument('--experiment_path', type=str, required=True, help='Path to the experiment results directory.')
    parser.add_argument('--output_dir', type=str, default=None, help="Optional: Base directory for evaluation results. Defaults to [experiment_path]/channel_debug_exps.")
    parser.add_argument('--fim_samples', type=int, default=1000, help="Number of samples for FIM calculation.")
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used.')
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument('--skip_eval', action='store_true', help="Skip evaluation, just collect and plot existing results.")
    cli_args = parser.parse_args()

    # ---- Load model and setup ----
    device = torch.device(f'cuda:{cli_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_data = load_model_for_utility_check(cli_args.experiment_path, device)
    model = model_data['model']
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']
    
    # Get channel estimation head
    if is_mtl:
        if 'channel_estimation' not in model['heads']:
            raise ValueError("Channel estimation head not found in MTL model")
        head = model['heads']['channel_estimation']
    else:
        if train_args.task != 'channel_estimation':
            raise ValueError("Model is not trained for channel estimation task")
        head = model['head']
    
    print(f"Channel estimation head loaded. Input dim: {head.input_dim}")

    activation_dir = os.path.join(cli_args.experiment_path, 'activations')
    partition_file = train_args.pkl_dataset_path
    
    with open(partition_file, 'rb') as handle:
        content = pickle.load(handle)
    train_files = content.get('train')
    test_files = content.get('test')

    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)
    test_dl = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False, num_workers=4)

    # --- Run Evaluations ---
    if not cli_args.skip_eval:
        base_output_dir, results_r2, results_mse = run_channel_evaluations(cli_args, head, test_dl, train_files, activation_dir)
    else:
        base_output_dir = cli_args.output_dir or os.path.join(cli_args.experiment_path, 'channel_debug_exps')
        # Load existing results if available
        results_json_path = os.path.join(base_output_dir, 'channel_estimation_results.json')
        if os.path.exists(results_json_path):
            with open(results_json_path, 'r') as f:
                aggregated_results = json.load(f)
                results_r2 = aggregated_results['r2_scores']
                results_mse = aggregated_results['mse_scores']
        else:
            print("No existing results found. Please run without --skip_eval first.")
            return

    # --- Plot Results ---
    plot_channel_results(base_output_dir, results_r2, results_mse)

if __name__ == '__main__':
    main() 