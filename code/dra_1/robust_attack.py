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
import json
from dotenv import load_dotenv
import wandb

# Add necessary paths to import other modules
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rep_lr')))

from dra_1.models import Decoder
from dra_1.py_datasets import ActivationDataset
from dra_1.inject_noise import inject_isotropic_noise, inject_nonisotropic_noise
from dra_1.model_loader import load_model_for_utility_check, load_training_args

from rep_lr.models import RFClassificationHead, CFOEstimationHead, ChannelEstimationHead, SimpleCFOEstimationHead, DirectCFOEstimationHead, CFOAdaptiveHead


def get_empirical_fim(head, data_loader, device, latent_dim):
    """Calculates the empirical Fisher Information Matrix for a given task head."""
    fim = torch.zeros((latent_dim, latent_dim), device=device)
    num_samples = 0
    head.to(device).eval()

    is_classification = isinstance(head, RFClassificationHead)
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    for batch in tqdm(data_loader, desc="Calculating FIM"):
        _, rf_labels, _, cfo_labels, _, ch_labels, activations, _ = batch
        
        activations = activations.squeeze(1).to(device)
        activations = activations.view(activations.size(0), -1)
        activations.requires_grad = True

        head.zero_grad()
        
        if is_classification:
            y_target = rf_labels.to(device)
            if y_target.dim() > 1: y_target = y_target.view(-1)
        elif isinstance(head, (CFOEstimationHead, SimpleCFOEstimationHead, DirectCFOEstimationHead, CFOAdaptiveHead)):
            y_target = cfo_labels.to(device).float()
        elif isinstance(head, ChannelEstimationHead):
            y_target = ch_labels.to(device)
            # --> THIS IS THE FIX: Remove the unnecessary dimension from the labels.
            if y_target.dim() == 4 and y_target.size(1) == 1:
                y_target = y_target.squeeze(1)
        else:
            continue

        y_hat = head(activations)
        if y_hat.dim() > 1 and y_target.dim() == 1 and not is_classification:
             y_hat = y_hat.squeeze(-1)

        loss = criterion(y_hat, y_target)
        loss.backward()
        
        if activations.grad is not None:
            J = activations.grad
            fim += J.T @ J
            num_samples += activations.size(0)

    if num_samples == 0:
        return fim
    return fim / num_samples


def main(args):
    """Main function to run the reconstruction attack."""
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Paths and Directories ---
    # Determine the output directory
    if args.output_dir:
        save_dir = args.output_dir
    else:
        # Fallback for standalone runs: create a directory based on parameters
        save_dir = os.path.join(
            args.experiment_path, 
            'attack_results_robust', 
            args.task, 
            args.noise_type, 
            f"leaked_frac_{str(args.leaked_fraction).replace('.', '_')}",
            f"noise_level_{str(args.noise_level).replace('.', '_')}"
        )
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in: {save_dir}")

    # Save args for reproducibility inside the final output directory
    with open(os.path.join(save_dir, 'attack_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Initialize wandb
    wandb.init(
        project="data-reconstruction-attack",
        config=args,
        name=f"attack_{args.task}_noise_{args.noise_type}_level_{args.noise_level}",
        dir=save_dir # Save wandb logs in the same directory
    )

    # --- Paths and Directories ---
    if hasattr(args, 'activations_path') and args.activations_path:
        activation_dir = args.activations_path
    else:
        activation_dir = os.path.join(args.experiment_path, 'activations')
    
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(f"Activations directory not found at {activation_dir}")


    # --- Load Data ---
    pkl_file_path = os.environ.get('PKL_FILE_PATH')
    if not pkl_file_path:
        raise ValueError("PKL_FILE_PATH environment variable not set.")
    
    partition_file = os.path.join(pkl_file_path, 'rf_partition_dict_0.5.pkl')
    with open(partition_file, 'rb') as f:
        partitions = pickle.load(f)
    
    train_files = partitions['train']
    val_files = partitions['val']
    test_files = partitions['test']

    # Sub-sample the training data based on the leaked fraction
    num_leaked_samples = int(len(train_files) * args.leaked_fraction)
    leaked_train_files = train_files[:num_leaked_samples]
    print(f"Using {len(leaked_train_files)} ({args.leaked_fraction*100}%) training samples for the attack.")


    train_dataset = ActivationDataset(activation_dir=activation_dir, file_list=leaked_train_files)
    val_dataset = ActivationDataset(activation_dir=activation_dir, file_list=val_files, test_mode=True)
    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- FIM Calculation (if needed) ---
    L, V = None, None
    if args.noise_type == 'nonisotropic':
        print("\nCalculating FIM for non-isotropic noise...")
        model_data = load_model_for_utility_check(args.experiment_path, device=device)
        train_args = model_data['train_args']

        # Use a subset of training data for FIM calculation
        fim_subset_indices = np.random.choice(len(train_dataset), size=args.fim_samples, replace=False)
        fim_dataset = Subset(train_dataset, fim_subset_indices)
        fim_loader = DataLoader(fim_dataset, batch_size=args.batch_size, shuffle=False)

        fim = torch.zeros((args.latent_dim, args.latent_dim), device=device)

        if model_data['is_mtl']:
            print("Model is MTL. Calculating and summing FIM for all tasks...")
            # For MTL, sum FIM across all tasks the model was trained on
            for task_name in train_args.task:
                print(f"  - Calculating FIM for task: {task_name}")
                head = model_data['model']['heads'][task_name]
                fim += get_empirical_fim(head, fim_loader, device, args.latent_dim)
        
        else: # Single-task model
            print("Model is single-task. Calculating FIM for the specific task...")
            model_task_name = train_args.task # e.g., 'cfo_estimation'
            
            task_map = {'rf': 'rf_fingerprinting', 'cfo': 'cfo_estimation', 'channel': 'channel_estimation'}
            attack_task_name = task_map[args.task]

            if model_task_name != attack_task_name:
                raise ValueError(
                    f"Attack task '{attack_task_name}' does not match the model's trained task '{model_task_name}'. "
                    f"Cannot calculate FIM for non-isotropic noise."
                )
            
            head = model_data['model']['head']
            fim = get_empirical_fim(head, fim_loader, device, args.latent_dim)

        print("Performing eigendecomposition of the final FIM...")
        L_e, V = torch.linalg.eigh(fim)
        L = torch.relu(L_e) # Ensure non-negative eigenvalues
        print("FIM calculation and eigendecomposition complete.")

    # --- Decoder Training ---
    print("\nStarting decoder training...")
    decoder = Decoder(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss() 
    
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_history = []

    for epoch in range(args.epochs):
        # Training loop
        decoder.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            rf_x, _, cfo_x, _, channel_x, _, activations, _ = batch
            activations = activations.squeeze(1).to(device)
            # --> THIS IS THE FIX: Flatten the latent vector before noise injection
            activations = activations.view(activations.size(0), -1)
            
            # Inject noise
            if args.noise_type == 'isotropic':
                activations = inject_isotropic_noise(activations, args.noise_level)
            elif args.noise_type == 'nonisotropic':
                activations = inject_nonisotropic_noise(activations, args.noise_level, L, V)
            
            optimizer.zero_grad()
            
            recons = decoder(activations, args.task)
            
            if args.task == 'rf':
                true_data = rf_x.squeeze(1).to(device)
            elif args.task == 'cfo':
                true_data = cfo_x.squeeze(1).to(device)
            elif args.task == 'channel':
                true_data = channel_x.squeeze(1).to(device)

            loss = criterion(recons, true_data)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loop
        decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                rf_x, _, cfo_x, _, channel_x, _, activations, _ = batch
                activations = activations.squeeze(1).to(device)
                # --> THIS IS THE FIX: Flatten the latent vector before noise injection
                activations = activations.view(activations.size(0), -1)

                if args.noise_type == 'isotropic':
                    activations = inject_isotropic_noise(activations, args.noise_level)
                elif args.noise_type == 'nonisotropic':
                    activations = inject_nonisotropic_noise(activations, args.noise_level, L, V)
                
                recons = decoder(activations, args.task)

                if args.task == 'rf':
                    true_data = rf_x.squeeze(1).to(device)
                elif args.task == 'cfo':
                    true_data = cfo_x.squeeze(1).to(device)
                elif args.task == 'channel':
                    true_data = channel_x.squeeze(1).to(device)

                loss = criterion(recons, true_data)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), os.path.join(save_dir, 'best_decoder.pt'))
            print("New best model saved.")
        else:
            patience_counter += 1

        # Record metrics for this epoch
        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss_so_far': best_val_loss
        })

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {patience_counter} epochs of no improvement.")
            break

    # --- Save Epoch History ---
    history_path = os.path.join(save_dir, 'epoch_history.json')
    with open(history_path, 'w') as f:
        json.dump(epoch_history, f, indent=4)
    print(f"Epoch history saved to {history_path}")

    # --- Final Evaluation on Test Set ---
    print("\nEvaluating on test set...")
    decoder.load_state_dict(torch.load(os.path.join(save_dir, 'best_decoder.pt')))
    decoder.eval()
    
    test_results = []
    total_test_mse = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            rf_x, _, cfo_x, _, channel_x, _, activations, filenames = batch
            activations = activations.squeeze(1).to(device)
            # --> THIS IS THE FIX: Flatten the latent vector before noise injection
            activations = activations.view(activations.size(0), -1)
            
            if args.noise_type == 'isotropic':
                activations = inject_isotropic_noise(activations, args.noise_level)
            elif args.noise_type == 'nonisotropic':
                activations = inject_nonisotropic_noise(activations, args.noise_level, L, V)

            recons = decoder(activations, args.task)
            
            if args.task == 'rf':
                true_data = rf_x.squeeze(1).to(device)
            elif args.task == 'cfo':
                true_data = cfo_x.squeeze(1).to(device)
            elif args.task == 'channel':
                true_data = channel_x.squeeze(1).to(device)

            mse_per_sample = ((recons - true_data)**2).mean(dim=[1, 2])
            total_test_mse += mse_per_sample.sum().item()

            # Store results for JSON output
            # Use .clone().detach() to create a deep copy of the tensor data,
            # preventing it from being overwritten in the next loop iteration.
            recons_copy = recons.clone().detach()
            true_data_copy = true_data.clone().detach()

            for j in range(len(filenames)):
                test_results.append({
                    'name': filenames[j],
                    f'true_{args.task}_x': true_data_copy[j].cpu().numpy().tolist(),
                    f'pred_{args.task}_x': recons_copy[j].cpu().numpy().tolist(),
                    'MSE': mse_per_sample[j].item()
                })

    avg_test_mse = total_test_mse / len(test_dataset)
    print(f"\nFinal Test MSE: {avg_test_mse:.6f}")
    wandb.log({"final_test_mse": avg_test_mse})

    # Save results to JSON
    results_path = os.path.join(save_dir, f"attack_{args.task}_results.json")
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {results_path}")

    # --- Visualize a few reconstructions ---
    print("Generating reconstruction plot...")
    originals = torch.tensor(np.array([res[f'true_{args.task}_x'] for res in test_results[:5]]))
    reconstructions = torch.tensor(np.array([res[f'pred_{args.task}_x'] for res in test_results[:5]]))

    # For the RF task, plot only a slice to avoid slow plotting with long sequences
    plot_slice = slice(None)  # Plot everything by default
    if args.task == 'rf':
        plot_slice = slice(0, 200)  # Plot only the first 200 samples for RF
        print("RF task detected, plotting a shorter slice of the signal for performance.")

    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    fig.suptitle(f'Reconstruction Samples - Task: {args.task.upper()}', fontsize=16)
    for i in range(5):
        orig_np = originals[i].numpy()
        recon_np = reconstructions[i].numpy()

        # Real Part
        axes[i, 0].plot(orig_np[0][plot_slice], label='Original')
        axes[i, 0].plot(recon_np[0][plot_slice], label='Reconstructed', linestyle='--')
        axes[i, 0].set_title(f"Sample {i+1} - Real Part")
        axes[i, 0].legend()

        # Imaginary Part
        axes[i, 1].plot(orig_np[1][plot_slice], label='Original')
        axes[i, 1].plot(recon_np[1][plot_slice], label='Reconstructed', linestyle='--')
        axes[i, 1].set_title(f"Sample {i+1} - Imaginary Part")
        axes[i, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(save_dir, "reconstruction_sample.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Reconstruction plot saved to {plot_path}")
    wandb.log({"reconstruction_samples": wandb.Image(plot_path)})

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Robust Data Reconstruction Attack Script")
    
    # Paths and identifiers
    parser.add_argument('--experiment_path', type=str, required=True, help="Path to the experiment directory.")
    parser.add_argument('--activations_path', type=str, default=None, help="Path to activations directory. If not provided, will use <experiment_path>/activations.")
    parser.add_argument('--output_dir', type=str, default=None, help="Path to the output directory for results. If not provided, a path will be constructed based on experiment parameters.")
    
    # Task and Noise
    parser.add_argument('--task', type=str, required=True, choices=['rf', 'cfo', 'channel'], help="Task to attack.")
    parser.add_argument('--noise_type', type=str, default='none', choices=['isotropic', 'nonisotropic', 'none'], help="Type of noise to inject.")
    parser.add_argument('--noise_level', type=float, default=0.0, help="Total variance of the noise.")
    parser.add_argument('--fim_samples', type=int, default=1000, help="Number of samples for FIM calculation.")
    parser.add_argument('--leaked_fraction', type=float, default=1.0, help="Fraction of the training dataset that is considered 'leaked' and used for the attack.")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=70, help="Max number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the decoder.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping.")

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=512, help="Dimension of the latent space.")
    
    args = parser.parse_args()
    
    main(args) 