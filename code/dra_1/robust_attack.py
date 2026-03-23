import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def run_reconstruction_attack(
    activation_dir,
    train_files,
    val_files,
    test_files,
    noise_type,
    noise_level,
    task,
    latent_dim,
    device,
    save_dir,
    L=None,
    V=None,
    lambda_factor=1e-5,
    clip_value=None,
    epochs=30,
    lr=1e-3,
    batch_size=64,
    patience=30,
    leaked_fraction=0.1,
    clip_grad_norm=1.0,
    use_wandb=False,
    wandb_project="data-reconstruction-attack-dp-runs-33",
) -> dict:
    """
    Train an adversary decoder and return reconstruction metrics.

    This function contains the core attack logic extracted from main() for
    programmatic reuse (e.g. from compare_methods.py).  All data-loading
    and FIM computation must be performed by the caller; L and V are the
    pre-computed FIM eigenvalues/eigenvectors needed for nonisotropic noise.

    Args:
        activation_dir   : directory containing pre-computed .pth activation files
        train_files      : list of training filenames (full partition, before leaked sub-sampling)
        val_files        : list of validation filenames
        test_files       : list of test filenames
        noise_type       : 'isotropic' | 'nonisotropic' | 'none'
        noise_level      : total noise variance (scalar)
        task             : 'rf' | 'cfo' | 'channel'
        latent_dim       : dimensionality of the latent space
        device           : torch.device
        save_dir         : directory to save decoder checkpoint and results
        L                : FIM eigenvalues tensor (required if noise_type='nonisotropic')
        V                : FIM eigenvectors tensor (required if noise_type='nonisotropic')
        lambda_factor    : FIM regularisation for nonisotropic noise
        clip_value       : optional activation norm clipping threshold
        epochs           : max training epochs
        lr               : learning rate
        batch_size       : data-loader batch size
        patience         : early-stopping patience
        leaked_fraction  : fraction of train_files used by the adversary
        clip_grad_norm   : gradient clipping norm (None to disable)
        use_wandb        : whether to log metrics to wandb
        wandb_project    : wandb project name (used only if use_wandb=True)

    Returns:
        dict with keys 'mse' (mean test MSE over all test samples)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Sub-sample training data to simulate limited adversary knowledge
    num_leaked = int(len(train_files) * leaked_fraction)
    leaked_train_files = train_files[:num_leaked]
    print(f"[Attack] Using {len(leaked_train_files)} / {len(train_files)} "
          f"({leaked_fraction*100:.0f}%) train samples.")

    train_dataset = ActivationDataset(activation_dir=activation_dir, file_list=leaked_train_files)
    val_dataset   = ActivationDataset(activation_dir=activation_dir, file_list=val_files,   test_mode=True)
    test_dataset  = ActivationDataset(activation_dir=activation_dir, file_list=test_files,  test_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

    # Decoder + optimiser
    decoder   = Decoder(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss     = float('inf')
    patience_counter  = 0
    epoch_history     = []

    def _apply_noise(act):
        if noise_type == 'isotropic':
            return inject_isotropic_noise(act, noise_level, clip_value=clip_value)
        elif noise_type == 'nonisotropic':
            return inject_nonisotropic_noise(act, noise_level, L, V, lambda_factor,
                                              clip_value=clip_value)
        elif clip_value is not None:
            return inject_isotropic_noise(act, 0.0, clip_value=clip_value)
        return act

    def _get_true(batch_tuple):
        rf_x, _, cfo_x, _, channel_x, _, _, _ = batch_tuple
        if task == 'rf':
            return rf_x.squeeze(1).to(device)
        elif task == 'cfo':
            return cfo_x.squeeze(1).to(device)
        return channel_x.squeeze(1).to(device)

    # Training loop
    for epoch in range(epochs):
        decoder.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            _, _, _, _, _, _, activations, _ = batch
            activations = activations.squeeze(1).to(device).view(-1, latent_dim)
            activations = _apply_noise(activations)

            true_data = _get_true(batch)
            optimizer.zero_grad()
            recons = decoder(activations, task)
            loss = criterion(recons, true_data)
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_grad_norm)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        decoder.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                _, _, _, _, _, _, activations, _ = batch
                activations = activations.squeeze(1).to(device).view(-1, latent_dim)
                activations = _apply_noise(activations)
                true_data = _get_true(batch)
                total_val_loss += criterion(decoder(activations, task), true_data).item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[Attack] Epoch {epoch+1}: train={avg_train_loss:.6f}  val={avg_val_loss:.6f}")

        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss,
                       "val_loss": avg_val_loss,
                       "learning_rate": optimizer.param_groups[0]['lr']})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), os.path.join(save_dir, 'best_decoder.pt'))
        else:
            patience_counter += 1

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        })

        if patience_counter >= patience:
            print(f"[Attack] Early stopping at epoch {epoch+1}.")
            break

    with open(os.path.join(save_dir, 'epoch_history.json'), 'w') as f:
        json.dump(epoch_history, f, indent=4)

    # Final evaluation on test set
    decoder.load_state_dict(torch.load(os.path.join(save_dir, 'best_decoder.pt'),
                                        map_location=device))
    decoder.eval()

    total_test_mse = 0.0
    n_test = 0
    test_results = []
    all_true_data, all_pred_data = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Attack] Testing", leave=False):
            rf_x, _, cfo_x, _, channel_x, _, activations, filenames = batch
            activations = activations.squeeze(1).to(device).view(-1, latent_dim)
            activations = _apply_noise(activations)

            true_data = _get_true(batch)
            recons = decoder(activations, task)

            mse_per_sample = ((recons - true_data) ** 2).mean(dim=[1, 2])
            total_test_mse += mse_per_sample.sum().item()
            n_test += true_data.size(0)

            for j, fname in enumerate(filenames):
                test_results.append({'name': fname, 'MSE': mse_per_sample[j].item()})

            all_true_data.append(true_data.cpu().numpy())
            all_pred_data.append(recons.cpu().numpy())

    avg_test_mse = total_test_mse / max(n_test, 1)
    print(f"[Attack] Final Test MSE: {avg_test_mse:.6f}")

    with open(os.path.join(save_dir, f"attack_{task}_results.json"), 'w') as f:
        json.dump(test_results, f, indent=4)

    if use_wandb:
        wandb.log({"final_test_mse": avg_test_mse})

    # Reconstruction plot
    all_true_np = np.concatenate(all_true_data, axis=0)
    all_pred_np = np.concatenate(all_pred_data, axis=0)
    originals       = torch.from_numpy(all_true_np[:5])
    reconstructions = torch.from_numpy(all_pred_np[:5])
    plot_slice = slice(0, 200) if task == 'rf' else slice(None)
    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    fig.suptitle(f'Reconstruction Samples - Task: {task.upper()}', fontsize=16)
    for i in range(min(5, len(originals))):
        for col, part_label in enumerate(['Real Part', 'Imaginary Part']):
            axes[i, col].plot(originals[i].numpy()[col][plot_slice], label='Original')
            axes[i, col].plot(reconstructions[i].numpy()[col][plot_slice],
                              label='Reconstructed', linestyle='--')
            axes[i, col].set_title(f"Sample {i+1} - {part_label}")
            axes[i, col].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(save_dir, "reconstruction_sample.png")
    plt.savefig(plot_path)
    plt.close()

    if use_wandb:
        wandb.log({"reconstruction_samples": wandb.Image(plot_path)})

    return {'mse': avg_test_mse}


def main(args):
    """Main function to run the reconstruction attack (standalone entry point)."""

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Output directory ---
    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = os.path.join(
            args.experiment_path,
            'attack_results_robust',
            args.task,
            args.noise_type,
            f"leaked_frac_{str(args.leaked_fraction).replace('.', '_')}",
            f"noise_level_{str(args.noise_level).replace('.', '_')}",
        )
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in: {save_dir}")

    with open(os.path.join(save_dir, 'attack_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    try:
        encoder_train_args = load_training_args(args.experiment_path)
    except FileNotFoundError as e:
        print(f"Warning: Could not load encoder training args: {e}")
        encoder_train_args = None

    # wandb init
    wandb_config = vars(args).copy()
    if encoder_train_args is not None:
        for key, value in vars(encoder_train_args).items():
            wandb_config[f"encoder_train_{key}"] = value

    wandb_name = f"attack_{args.task}_noise_{args.noise_type}_level_{args.noise_level}"
    if args.noise_type == 'nonisotropic':
        wandb_name += f"_lambda_{args.lambda_factor}"
    if args.clip_value is not None:
        wandb_name += f"_clip_{args.clip_value}"
    wandb.init(
        project="data-reconstruction-attack-dp-runs-33",
        config=wandb_config,
        name=wandb_name,
    )

    # --- Activation directory ---
    if hasattr(args, 'activations_path') and args.activations_path:
        activation_dir = args.activations_path
    else:
        activation_dir = os.path.join(args.experiment_path, 'activations')
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(f"Activations directory not found at {activation_dir}")

    # --- Load partition ---
    pkl_file_path = os.environ.get('PKL_FILE_PATH')
    if not pkl_file_path:
        raise ValueError("PKL_FILE_PATH environment variable not set.")
    partition_file = os.path.join(pkl_file_path, 'rf_partition_dict_0.5.pkl')
    with open(partition_file, 'rb') as f:
        partitions = pickle.load(f)
    train_files = partitions['train']
    val_files   = partitions['val']
    test_files  = partitions['test']

    # --- FIM (if needed) ---
    L, V = None, None
    if args.noise_type == 'nonisotropic':
        print("\nCalculating FIM for non-isotropic noise...")
        model_data = load_model_for_utility_check(args.experiment_path, device=device)
        train_args_enc = model_data['train_args']

        train_full_dataset = ActivationDataset(activation_dir=activation_dir, file_list=train_files)
        fim_subset_indices = np.random.choice(len(train_full_dataset), size=args.fim_samples, replace=False)
        fim_dataset = Subset(train_full_dataset, fim_subset_indices)
        fim_loader  = DataLoader(fim_dataset, batch_size=args.batch_size, shuffle=False)

        fim = torch.zeros((args.latent_dim, args.latent_dim), device=device)

        if model_data['is_mtl']:
            print("MTL model: summing FIM over all tasks...")
            for task_name in train_args_enc.task:
                print(f"  - FIM for {task_name}")
                fim += get_empirical_fim(
                    model_data['model']['heads'][task_name],
                    fim_loader, device, args.latent_dim
                )
        else:
            task_map = {
                'rf': 'rf_fingerprinting',
                'cfo': 'cfo_estimation',
                'channel': 'channel_estimation',
            }
            attack_task_name = task_map[args.task]
            if train_args_enc.task != attack_task_name:
                raise ValueError(
                    f"Attack task '{attack_task_name}' does not match model task "
                    f"'{train_args_enc.task}'."
                )
            fim = get_empirical_fim(model_data['model']['head'], fim_loader, device, args.latent_dim)

        trace_fim = torch.trace(fim)
        if trace_fim > 1e-10:
            fim = fim / trace_fim
            print(f"  FIM normalised by trace: {trace_fim:.3e}")
        else:
            print("  Warning: FIM trace near zero; skipping normalisation.")

        L_e, V = torch.linalg.eigh(fim)
        L = torch.relu(L_e)
        print("FIM eigendecomposition complete.")

    # --- Delegate to run_reconstruction_attack ---
    metrics = run_reconstruction_attack(
        activation_dir=activation_dir,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        task=args.task,
        latent_dim=args.latent_dim,
        device=device,
        save_dir=save_dir,
        L=L,
        V=V,
        lambda_factor=args.lambda_factor,
        clip_value=args.clip_value,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        leaked_fraction=args.leaked_fraction,
        clip_grad_norm=args.clip_grad_norm,
        use_wandb=True,
        wandb_project="data-reconstruction-attack-dp-runs-33",
    )

    print(f"\nAttack complete. Test MSE: {metrics['mse']:.6f}")
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
    parser.add_argument('--epochs', type=int, default=30, help="Max number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the decoder.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--patience', type=int, default=30, help="Patience for early stopping.")
    parser.add_argument('--use_lr_scheduler', action='store_true', help="Enable learning rate scheduler (ReduceLROnPlateau).")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help="Optimizer to use for training.")
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help="Maximum norm for gradient clipping. If None, no clipping is performed.")
    parser.add_argument('--clip_value', type=float, default=None, help="Optional norm clip value for activations.")


    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=512, help="Dimension of the latent space.")
    parser.add_argument('--lambda_factor', type=float, default=1e-5, help="Lambda factor for FIM regularization in non-isotropic noise.")
    
    args = parser.parse_args()
    
    main(args) 