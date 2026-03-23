import os
import sys
import json
import argparse
import random
import tempfile
import shutil
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pickle
import wandb

# Paths for cross-package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rep_lr')))

from dra_1.model_loader import (
    load_training_args,
    create_projection_layer,
    create_encoder,
)
from dra_1.models import Decoder
from dra_1.py_datasets import ActivationDataset
from rep_lr.py_datasets import (
    TrainDataset,
    TrainDatasetRFixed,
    collate_fn_rf_fixed,
)
from rep_lr.models import (
    RFClassificationHead,
    ChannelEstimationHead,
    CFOEstimationHead,
)
from sklearn.metrics import mean_squared_error, r2_score


# Allowed experiment task folders (same filter pattern as utility sweeps)
# ALLOWED_TASK_DIRS = {
#     "cfo", "rf", "rf_channel",
#     "cfo_channel", "rf_cfo",
#     "channel", "rf_cfo_channel"
# }

ALLOWED_TASK_DIRS = {
"rf_cfo_channel"
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_random_encoder_stack(train_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build a randomly initialized projection + encoder stack that matches the
    architecture described by train_args. No trained weights are loaded.
    Returns a dict mirroring model components as used elsewhere in this repo.
    """
    is_mtl = getattr(train_args, 'mtl', False)
    if is_mtl:
        proj_dict = nn.ModuleDict()
        for task in train_args.task:
            proj_dict[task] = create_projection_layer(task, train_args)
        encoder = create_encoder(train_args)
        return {'projections': proj_dict, 'encoder': encoder, 'is_mtl': True}
    else:
        task = train_args.task
        projection = create_projection_layer(task, train_args)
        encoder = create_encoder(train_args)
        return {'projection': projection, 'encoder': encoder, 'is_mtl': False, 'task': task}


def forward_random_encoder(
    encoder_stack: Dict[str, Any],
    train_args: argparse.Namespace,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> torch.Tensor:
    """
    Given a batch from TrainDataset/TrainDatasetRFixed and a randomly initialized
    encoder stack, compute encoder activations.
    Returns tensor of shape (B, 2, d2).
    """
    is_mtl = encoder_stack.get('is_mtl', False)

    if is_mtl:
        # Batch tuple: rf_inputs, rf_labels, cfo_inputs, cfo_labels, channel_inputs, channel_labels, file_paths
        rf_inputs, _, cfo_inputs, _, channel_inputs, _, _ = batch
        # Handle sliding-window eval shapes: (B, num_slices, 2, L) -> take first slice
        if rf_inputs.dim() == 4 and rf_inputs.size(1) > 0:
            rf_inputs = rf_inputs[:, 0, :, :]
        if cfo_inputs.dim() == 4 and cfo_inputs.size(1) > 0:
            cfo_inputs = cfo_inputs[:, 0, :, :]
        if channel_inputs.dim() == 4 and channel_inputs.size(1) > 0:
            channel_inputs = channel_inputs[:, 0, :, :]
        task_data_map = {
            'rf_fingerprinting': rf_inputs.to(device).float(),
            'cfo_estimation': cfo_inputs.to(device).float(),
            'channel_estimation': channel_inputs.to(device).float(),
        }
        projected_tensors = []
        for task_name in train_args.task:
            inputs = task_data_map[task_name]
            proj = encoder_stack['projections'][task_name](inputs)
            projected_tensors.append(proj)

        fusion_type = getattr(train_args, 'fusion_type', 'sum')
        if fusion_type == 'concat':
            projected_input = torch.cat(projected_tensors, dim=2)
        elif fusion_type == 'depth_concat':
            projected_input = torch.cat(projected_tensors, dim=1)
        else:
            projected_input = torch.sum(torch.stack(projected_tensors), dim=0)

        encoded = encoder_stack['encoder'](projected_input)
        return encoded

    # Single task
    rf_inputs, _, cfo_inputs, _, channel_inputs, _, _ = batch
    if rf_inputs.dim() == 4 and rf_inputs.size(1) > 0:
        rf_inputs = rf_inputs[:, 0, :, :]
    if cfo_inputs.dim() == 4 and cfo_inputs.size(1) > 0:
        cfo_inputs = cfo_inputs[:, 0, :, :]
    if channel_inputs.dim() == 4 and channel_inputs.size(1) > 0:
        channel_inputs = channel_inputs[:, 0, :, :]
    task_name = train_args.task
    if task_name == 'rf_fingerprinting':
        inputs = rf_inputs.to(device).float()
    elif task_name == 'cfo_estimation':
        inputs = cfo_inputs.to(device).float()
    elif task_name == 'channel_estimation':
        inputs = channel_inputs.to(device).float()
    else:
        raise ValueError(f"Unknown task in train_args: {task_name}")

    proj = encoder_stack['projection'](inputs)
    encoded = encoder_stack['encoder'](proj)
    return encoded


def get_partitions_from_args_or_env(train_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load the partitions file (train/val/test lists) and aux stats from
    either train_args.pkl_dataset_path or PKL_FILE_PATH env var.
    """
    if hasattr(train_args, 'pkl_dataset_path') and train_args.pkl_dataset_path:
        partition_file = train_args.pkl_dataset_path
    else:
        pkl_file_path_env = os.environ.get('PKL_FILE_PATH')
        if not pkl_file_path_env:
            raise ValueError("Neither train_args.pkl_dataset_path nor PKL_FILE_PATH env var is set.")
        partition_file = os.path.join(pkl_file_path_env, 'rf_partition_dict_0.5.pkl')

    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file not found at {partition_file}")

    with open(partition_file, 'rb') as f:
        partitions = pickle.load(f)
    return partitions


def make_dataloaders_for_files(
    files: List[str],
    train_args: argparse.Namespace,
    batch_size: int,
    num_workers: int,
    max_cfo: Any,
    mean_cfo: Any,
    std_cfo: Any,
    shuffle: bool,
    test_mode: bool,
) -> DataLoader:
    """
    Construct a DataLoader over raw data files using TrainDataset/TrainDatasetRFixed.
    """
    ID_class_dict = {f'Radio{i}': i for i in range(16)}

    if getattr(train_args, 'rf_fixed', False):
        dataset = TrainDatasetRFixed(
            files, ID_class_dict, train_args, max_cfo, mean_cfo, std_cfo, test_mode=test_mode,
            rf_begin_idx=getattr(train_args, 'rf_begin_idx', 0)
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_rf_fixed
        )
    else:
        dataset = TrainDataset(files, ID_class_dict, train_args, max_cfo, mean_cfo, std_cfo, test_mode=test_mode)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    return loader


def extract_random_activations(
    encoder_stack: Dict[str, Any],
    train_args: argparse.Namespace,
    device: torch.device,
    partitions: Dict[str, Any],
    temp_activation_dir: str,
    num_workers: int = 4,
) -> None:
    """
    Extract activations from random encoder for all data (train+val+test) and save to temp dir.
    This mirrors extract_activations.py but uses a random encoder.
    """
    max_cfo = partitions.get('max_cfo')
    mean_cfo = partitions.get('mean_cfo')
    std_cfo = partitions.get('std_cfo')
    
    # Combine all data
    all_files = partitions['train'] + partitions['val'] + partitions['test']
    
    # Create dataloader with batch_size=1 to mirror extract_activations.py
    data_loader = make_dataloaders_for_files(
        all_files, train_args, batch_size=1, num_workers=num_workers,
        max_cfo=max_cfo, mean_cfo=mean_cfo, std_cfo=std_cfo,
        shuffle=False, test_mode=False
    )
    
    os.makedirs(temp_activation_dir, exist_ok=True)
    print(f"Extracting random activations to: {temp_activation_dir}")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting random activations"):
            if batch[0] is None:
                continue
            rf_x, rf_y, cfo_x, cfo_y, ch_x, ch_y, file_paths = batch
            
            # Handle multiple slices from rf_fixed
            is_rf_fixed = getattr(train_args, 'rf_fixed', False)
            if is_rf_fixed and rf_x.shape[0] > 1:
                num_slices = rf_x.shape[0]
                filename = os.path.basename(file_paths[0])
                for slice_idx in range(num_slices):
                    slice_filename = filename.replace('.mat', f'_slice{slice_idx}.pth')
                    save_path = os.path.join(temp_activation_dir, slice_filename)
                    
                    # Extract slice data
                    rf_slice = rf_x[slice_idx:slice_idx+1]
                    if rf_y.numel() > 1:
                        rf_label_slice = rf_y[slice_idx:slice_idx+1]
                    else:
                        rf_label_slice = rf_y
                    cfo_slice = cfo_x[:1]
                    cfo_label_slice = cfo_y[:1] if cfo_y.numel() > 1 else cfo_y
                    channel_slice = ch_x[:1]
                    channel_label_slice = ch_y[:1] if ch_y.dim() > 0 and ch_y.shape[0] > 1 else ch_y
                    
                    # Create temporary batch for forward pass
                    temp_batch = (rf_slice, rf_label_slice, cfo_slice, cfo_label_slice, channel_slice, channel_label_slice, [file_paths[0]])
                    activation = forward_random_encoder(encoder_stack, train_args, temp_batch, device)
                    
                    data_to_save = {
                        'activation': activation.detach().cpu(),
                        'filename': slice_filename,
                        'RF_X': rf_slice.detach().cpu(),
                        'CFO_X': cfo_slice.detach().cpu(),
                        'Channel_X': channel_slice.detach().cpu(),
                        'rf_label': rf_label_slice.detach().cpu(),
                        'cfo_label': cfo_label_slice.detach().cpu(),
                        'channel_label': channel_label_slice.detach().cpu()
                    }
                    torch.save(data_to_save, save_path, _use_new_zipfile_serialization=False)
            else:
                # Regular single sample
                filename = os.path.basename(file_paths[0])
                save_path = os.path.join(temp_activation_dir, filename.replace('.mat', '.pth'))
                
                activation = forward_random_encoder(encoder_stack, train_args, batch, device)
                
                # Handle sliding window shapes for saving
                if rf_x.dim() == 4:
                    rf_x = rf_x[:, 0, :, :]
                if cfo_x.dim() == 4:
                    cfo_x = cfo_x[:, 0, :, :]
                if ch_x.dim() == 4:
                    ch_x = ch_x[:, 0, :, :]
                
                data_to_save = {
                    'activation': activation.detach().cpu(),
                    'filename': filename,
                    'RF_X': rf_x.detach().cpu(),
                    'CFO_X': cfo_x.detach().cpu(),
                    'Channel_X': ch_x.detach().cpu(),
                    'rf_label': rf_y.detach().cpu(),
                    'cfo_label': cfo_y.detach().cpu(),
                    'channel_label': ch_y.detach().cpu()
                }
                torch.save(data_to_save, save_path, _use_new_zipfile_serialization=False)
    
    print(f"Random activation extraction complete. Saved to {temp_activation_dir}")


def train_decoder_with_cached_activations(
    activation_dir: str,
    train_args: argparse.Namespace,
    device: torch.device,
    partitions: Dict[str, Any],
    tasks_to_reconstruct: List[str],
    save_dir: str,
    leaked_fraction: float = 1.0,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 30,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """
    Train a decoder to reconstruct original signals from cached random-encoder activations.
    Returns average test MSE per task.
    """
    train_files = partitions['train']
    val_files = partitions['val']
    test_files = partitions['test']

    # Select leaked subset for training the decoder
    num_leaked = max(1, int(len(train_files) * leaked_fraction))
    leaked_train_files = train_files[:num_leaked]

    # Create ActivationDataset loaders
    train_dataset = ActivationDataset(activation_dir=activation_dir, file_list=leaked_train_files, test_mode=False)
    val_dataset = ActivationDataset(activation_dir=activation_dir, file_list=val_files, test_mode=True)
    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Map pretty task keys to short names for Decoder
    task_key_map = {
        'rf_fingerprinting': 'rf',
        'cfo_estimation': 'cfo',
        'channel_estimation': 'channel',
    }

    results: Dict[str, Any] = {}

    # Train separate decoder per task for clarity and to avoid interference
    for pretty_task in tasks_to_reconstruct:
        short_task = task_key_map[pretty_task]

        decoder = Decoder(latent_dim=2 * train_args.d2).to(device)
        optimizer = optim.Adam(decoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        task_save_dir = os.path.join(save_dir, f"decoder_{short_task}")
        os.makedirs(task_save_dir, exist_ok=True)

        # Train
        for epoch in range(epochs):
            decoder.train()
            total_train = 0.0
            for batch in tqdm(train_loader, desc=f"[Decoder-{short_task}] Train {epoch+1}/{epochs}"):
                rf_x, _, cfo_x, _, ch_x, _, activations, _ = batch
                activations = activations.squeeze(1).to(device)
                z = activations.view(activations.size(0), -1)  # (B, 2*d2)

                if short_task == 'rf':
                    true_data = rf_x.squeeze(1).to(device).float()
                elif short_task == 'cfo':
                    true_data = cfo_x.squeeze(1).to(device).float()
                else:
                    true_data = ch_x.squeeze(1).to(device).float()

                optimizer.zero_grad()
                recons = decoder(z, short_task)
                loss = criterion(recons, true_data)
                loss.backward()
                optimizer.step()
                total_train += loss.item()

            avg_train = total_train / max(1, len(train_loader))

            # Validation
            decoder.eval()
            total_val = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"[Decoder-{short_task}] Val {epoch+1}/{epochs}"):
                    rf_x, _, cfo_x, _, ch_x, _, activations, _ = batch
                    activations = activations.squeeze(1).to(device)
                    z = activations.view(activations.size(0), -1)
                    if short_task == 'rf':
                        true_data = rf_x.squeeze(1).to(device).float()
                    elif short_task == 'cfo':
                        true_data = cfo_x.squeeze(1).to(device).float()
                    else:
                        true_data = ch_x.squeeze(1).to(device).float()
                    recons = decoder(z, short_task)
                    loss = criterion(recons, true_data)
                    total_val += loss.item()
            avg_val = total_val / max(1, len(val_loader))

            # wandb logging per-epoch for decoder
            wandb.log({
                f"decoder/{short_task}/epoch": epoch + 1,
                f"decoder/{short_task}/train_loss": avg_train,
                f"decoder/{short_task}/val_loss": avg_val,
            })

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(decoder.state_dict(), os.path.join(task_save_dir, 'best_decoder.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Test
        decoder.load_state_dict(torch.load(os.path.join(task_save_dir, 'best_decoder.pt'), map_location=device))
        decoder.eval()
        total_test_mse = 0.0
        num_samples = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Decoder-{short_task}] Test"):
                rf_x, _, cfo_x, _, ch_x, _, activations, _ = batch
                activations = activations.squeeze(1).to(device)
                z = activations.view(activations.size(0), -1)
                if short_task == 'rf':
                    true_data = rf_x.squeeze(1).to(device).float()
                elif short_task == 'cfo':
                    true_data = cfo_x.squeeze(1).to(device).float()
                else:
                    true_data = ch_x.squeeze(1).to(device).float()
                recons = decoder(z, short_task)
                # Mean MSE per-sample over channel and length dims
                mse_per_sample = ((recons - true_data) ** 2).mean(dim=[1, 2])
                total_test_mse += mse_per_sample.sum().item()
                num_samples += mse_per_sample.numel()
        avg_test_mse = total_test_mse / max(1, num_samples)
        wandb.log({f"decoder/{short_task}/test_mse": avg_test_mse})
        results[short_task] = {'reconstruction_mse': avg_test_mse}

    return results


def create_head_for_neuracrypt(task_name: str, train_args: argparse.Namespace) -> nn.Module:
    """
    Create a downstream head for utility training/eval. For CFO, explicitly use
    CFOEstimationHead (feature-based), ignoring any 'direct_cfo' flags, since
    this baseline evaluates features from the random encoder.
    """
    input_dim = 2 * train_args.d2
    if task_name == 'rf_fingerprinting':
        return RFClassificationHead(input_dim=input_dim, num_classes=16, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)
    elif task_name == 'channel_estimation':
        return ChannelEstimationHead(input_dim=input_dim, hidden_dim=train_args.head_hidden_dim, output_length=52, dropout=train_args.dropout)
    elif task_name == 'cfo_estimation':
        return CFOEstimationHead(input_dim=input_dim, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)
    else:
        raise ValueError(f"Unknown task name for head creation: {task_name}")


def train_heads_with_cached_activations(
    activation_dir: str,
    train_args: argparse.Namespace,
    device: torch.device,
    partitions: Dict[str, Any],
    tasks_to_train: List[str],
    save_dir: str,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """
    Train downstream task heads on cached random encoder features and evaluate utility.
    Returns metrics dict per task: accuracy for RF; mse and r2 for CFO/Channel.
    """
    train_files = partitions['train']
    val_files = partitions['val']
    test_files = partitions['test']

    train_dataset = ActivationDataset(activation_dir=activation_dir, file_list=train_files, test_mode=False)
    val_dataset = ActivationDataset(activation_dir=activation_dir, file_list=val_files, test_mode=True)
    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    results: Dict[str, Any] = {}

    for pretty_task in tasks_to_train:
        head = create_head_for_neuracrypt(pretty_task, train_args).to(device)

        if pretty_task == 'rf_fingerprinting':
            criterion = nn.CrossEntropyLoss()
        else:
            # For regression tasks (CFO, Channel), use MSE
            criterion = nn.MSELoss()

        optimizer = optim.Adam(head.parameters(), lr=lr)
        best_val = float('inf')
        head_save_dir = os.path.join(save_dir, f"head_{pretty_task}")
        os.makedirs(head_save_dir, exist_ok=True)

        # Train
        for epoch in range(epochs):
            head.train()
            total_train = 0.0
            for batch in tqdm(train_loader, desc=f"[Head-{pretty_task}] Train {epoch+1}/{epochs}"):
                _, rf_y, _, cfo_y, _, ch_y, activations, _ = batch
                activations = activations.squeeze(1).to(device)  # (B, 2, d2)
                features = activations  # Head implementations expect (B, 2, d2)

                optimizer.zero_grad()
                outputs = head(features)

                if pretty_task == 'rf_fingerprinting':
                    labels = rf_y.to(device).view(-1)
                    loss = criterion(outputs, labels)
                elif pretty_task == 'cfo_estimation':
                    labels = cfo_y.to(device).float().view(-1)
                    outputs = outputs.view(-1)
                    loss = criterion(outputs, labels)
                else:  # channel_estimation
                    labels = ch_y.to(device).float()
                    if labels.dim() == 4:
                        labels = labels.squeeze(1)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                total_train += loss.item()

            avg_train = total_train / max(1, len(train_loader))

            # Validation
            head.eval()
            total_val = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"[Head-{pretty_task}] Val {epoch+1}/{epochs}"):
                    _, rf_y, _, cfo_y, _, ch_y, activations, _ = batch
                    activations = activations.squeeze(1).to(device)
                    outputs = head(activations)
                    if pretty_task == 'rf_fingerprinting':
                        labels = rf_y.to(device).view(-1)
                        loss = criterion(outputs, labels)
                    elif pretty_task == 'cfo_estimation':
                        labels = cfo_y.to(device).float().view(-1)
                        outputs = outputs.view(-1)
                        loss = criterion(outputs, labels)
                    else:
                        labels = ch_y.to(device).float()
                        if labels.dim() == 4:
                            labels = labels.squeeze(1)
                        loss = criterion(outputs, labels)
                    total_val += loss.item()

            avg_val = total_val / max(1, len(val_loader))
            # wandb logging per-epoch for heads
            wandb.log({
                f"head/{pretty_task}/epoch": epoch + 1,
                f"head/{pretty_task}/train_loss": avg_train,
                f"head/{pretty_task}/val_loss": avg_val,
            })
            if avg_val < best_val:
                best_val = avg_val
                torch.save(head.state_dict(), os.path.join(head_save_dir, 'best_head.pt'))

        # Test/Eval
        head.load_state_dict(torch.load(os.path.join(head_save_dir, 'best_head.pt'), map_location=device))
        head.eval()

        y_true_list: List[np.ndarray] = []
        y_pred_list: List[np.ndarray] = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Head-{pretty_task}] Test"):
                _, rf_y, _, cfo_y, _, ch_y, activations, _ = batch
                activations = activations.squeeze(1).to(device)
                outputs = head(activations)

                if pretty_task == 'rf_fingerprinting':
                    labels = rf_y.to(device).view(-1)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                elif pretty_task == 'cfo_estimation':
                    labels = cfo_y.to(device).float().view(-1)
                    outputs = outputs.view(-1)
                    y_true_list.append(labels.detach().cpu().numpy())
                    y_pred_list.append(outputs.detach().cpu().numpy())
                else:
                    labels = ch_y.to(device).float()
                    if labels.dim() == 4:
                        labels = labels.squeeze(1)
                    y_true_list.append(labels.detach().cpu().numpy())
                    y_pred_list.append(outputs.detach().cpu().numpy())

        # Metrics
        if pretty_task == 'rf_fingerprinting':
            acc = correct / max(1, total)
            results['rf_fingerprinting'] = {'accuracy': acc}
        else:
            y_true = np.concatenate(y_true_list, axis=0)
            y_pred = np.concatenate(y_pred_list, axis=0)
            if pretty_task == 'channel_estimation':
                r2 = r2_score(y_true.reshape(len(y_true), -1), y_pred.reshape(len(y_pred), -1))
                mse = mean_squared_error(y_true.reshape(len(y_true), -1), y_pred.reshape(len(y_pred), -1))
            else:
                r2 = r2_score(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
            results[pretty_task] = {'r2': float(r2), 'mse': float(mse)}

    return results


def determine_tasks_from_train_args(train_args: argparse.Namespace) -> List[str]:
    """
    Return list of canonical task names present in this experiment based on train_args.
    """
    if getattr(train_args, 'mtl', False):
        return list(train_args.task)
    return [train_args.task]


def main():
    parser = argparse.ArgumentParser(description="NeuraCrypt baseline: random encoder features for reconstruction and utility.")
    parser.add_argument("--experiment_path", type=str, required=True, help="Path to main run directory (e.g., .../run_1).")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Decoder (reconstruction) training
    parser.add_argument("--leaked_fraction", type=float, default=1.0, help="Fraction of train set used to train decoder.")
    parser.add_argument("--decoder_epochs", type=int, default=30)
    parser.add_argument("--decoder_lr", type=float, default=1e-3)
    parser.add_argument("--decoder_batch_size", type=int, default=64)
    parser.add_argument("--decoder_patience", type=int, default=30)
    # Head (utility) training
    parser.add_argument("--head_epochs", type=int, default=10)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--head_batch_size", type=int, default=64)
    # Optional filter override
    parser.add_argument("--allowed_tasks_all", action="store_true", help="If set, process all subdirectories, not just filtered set.")
    # wandb
    parser.add_argument("--wandb_project", type=str, default="neuracrypt-baseline", help="W&B project name.")
    parser.add_argument("--wandb_group", type=str, default="neuracrypt", help="W&B group name.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    run_root = args.experiment_path
    if not os.path.isdir(run_root):
        print(f"Error: experiment_path is not a directory: {run_root}")
        return

    # Determine which subdirectories to process
    all_subdirs = [d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d))]
    if args.allowed_tasks_all:
        task_dirs = all_subdirs
    else:
        task_dirs = [d for d in all_subdirs if d in ALLOWED_TASK_DIRS]
    print(f"Using task directories: {task_dirs}")
    if not task_dirs:
        print("No matching task directories to process.")
        return

    for task_dir in task_dirs:
        task_dir_path = os.path.join(run_root, task_dir)
        # Find the inner timestamped experiment folder
        exp_folder = None
        for folder in os.listdir(task_dir_path):
            if os.path.isdir(os.path.join(task_dir_path, folder)):
                exp_folder = folder
                break
        if exp_folder is None:
            print(f"Skipping empty: {task_dir_path}")
            continue

        full_exp_path = os.path.join(task_dir_path, exp_folder)
        print(f"\n--- Processing experiment: {full_exp_path} ---")

        # Load training args (architecture spec), build random encoder stack
        train_args = load_training_args(full_exp_path)
        encoder_stack = build_random_encoder_stack(train_args)
        # Move modules to device and set to eval for feature extraction, but heads/decoder will be in train
        if encoder_stack.get('is_mtl', False):
            encoder_stack['projections'].to(device).eval()
            encoder_stack['encoder'].to(device).eval()
        else:
            encoder_stack['projection'].to(device).eval()
            encoder_stack['encoder'].to(device).eval()

        # Load partitions
        partitions = get_partitions_from_args_or_env(train_args)
        # Identify tasks present for this experiment
        present_tasks = determine_tasks_from_train_args(train_args)

        # Create output directory
        out_dir = os.path.join(full_exp_path, "neuracrypt_baseline")
        os.makedirs(out_dir, exist_ok=True)

        # 0) Extract random encoder activations to temporary directory
        temp_activation_dir = tempfile.mkdtemp(prefix="neuracrypt_activations_")
        print(f"Created temporary activation directory: {temp_activation_dir}")
        try:
            extract_random_activations(
                encoder_stack=encoder_stack,
                train_args=train_args,
                device=device,
                partitions=partitions,
                temp_activation_dir=temp_activation_dir,
                num_workers=4,
            )

            # Initialize wandb per sub-experiment
            run_name = f"neuracrypt_{task_dir}_{exp_folder}"
            # Build config: CLI args + encoder train args (prefixed)
            wandb_config = {
                "seed": args.seed,
                "gpu_id": args.gpu_id,
                "leaked_fraction": args.leaked_fraction,
                "decoder_epochs": args.decoder_epochs,
                "decoder_lr": args.decoder_lr,
                "decoder_batch_size": args.decoder_batch_size,
                "decoder_patience": args.decoder_patience,
                "head_epochs": args.head_epochs,
                "head_lr": args.head_lr,
                "head_batch_size": args.head_batch_size,
                "present_tasks": present_tasks,
                "allowed_tasks_all": args.allowed_tasks_all,
            }
            for k, v in vars(train_args).items():
                wandb_config[f"encoder_train_{k}"] = v
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=run_name,
                config=wandb_config,
                tags=["neuracrypt", "random_encoder"] + (["mtl"] if getattr(train_args, "mtl", False) else ["single_task"]),
            )

            # 1) Train downstream head(s) with cached activations and evaluate utility (FIRST)
            utility_results = train_heads_with_cached_activations(
                activation_dir=temp_activation_dir,
                train_args=train_args,
                device=device,
                partitions=partitions,
                tasks_to_train=present_tasks,
                save_dir=out_dir,
                batch_size=args.head_batch_size,
                epochs=args.head_epochs,
                lr=args.head_lr,
                num_workers=8,
            )

            # 2) Train decoder(s) on leaked set using cached activations (SECOND)
            recon_results = train_decoder_with_cached_activations(
                activation_dir=temp_activation_dir,
                train_args=train_args,
                device=device,
                partitions=partitions,
                tasks_to_reconstruct=present_tasks,
                save_dir=out_dir,
                leaked_fraction=args.leaked_fraction,
                batch_size=args.decoder_batch_size,
                epochs=args.decoder_epochs,
                lr=args.decoder_lr,
                patience=args.decoder_patience,
                num_workers=8,
            )
        finally:
            # Clean up temp activations
            if os.path.exists(temp_activation_dir):
                shutil.rmtree(temp_activation_dir)
                print(f"Cleaned up temporary activation directory: {temp_activation_dir}")

        # 3) Save combined results
        combined: Dict[str, Any] = {'reconstruction': recon_results, 'utility': utility_results}
        # Log combined to wandb
        # Flatten for wandb-friendly keys
        log_payload = {}
        for k, v in recon_results.items():
            for mk, mv in v.items():
                log_payload[f"reconstruction/{k}/{mk}"] = mv
        for k, v in utility_results.items():
            for mk, mv in v.items():
                log_payload[f"utility/{k}/{mk}"] = mv
        if log_payload:
            wandb.log(log_payload)

        results_path = os.path.join(out_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=4)
        print(f"Saved NeuraCrypt baseline results to: {results_path}")
        wandb.finish()


if __name__ == "__main__":
    main()


