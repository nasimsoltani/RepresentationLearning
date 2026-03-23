#!/usr/bin/env python3
"""
Train a distance classification model from activations.

This script loads activations and trains a classification head to predict
distance from the activation representations. Distance is extracted from
filenames like: RFfingerprinting_run1_Radio0_14ft_0.pth where 14 is the distance.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle
import re
from tqdm import tqdm
import json
from dotenv import load_dotenv
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add necessary paths to import other modules
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rep_lr')))

from dra_1.py_datasets import ActivationDataset
from dra_1.inject_noise import inject_nonisotropic_noise
from rep_lr.models import RFClassificationHead


class DistanceDataset(Dataset):
    """
    Dataset class that reads activations and extracts distance labels from filenames.
    
    Filenames are expected to be in format: RFfingerprinting_run1_Radio0_14ft_0.pth
    where 14 is the distance value.
    """
    
    def __init__(self, activation_dir, file_list, test_mode=False):
        """
        Args:
            activation_dir (string): Directory with all the activation .pth files.
            file_list (list): List of filenames to include in this dataset slice.
            test_mode (bool): In test mode, we don't need to load all data, just what's needed for one sample.
        """
        self.activation_dir = activation_dir
        self.file_list = list(file_list)
        self.test_mode = test_mode
        
        if not os.path.isdir(self.activation_dir):
            raise ValueError(f"Activation directory not found: {self.activation_dir}")
        
        # Extract distances from filenames and build mapping
        self.activation_map = {}
        self.distances = []
        self.unique_distances = set()
        
        # Pattern to extract distance: RFfingerprinting_run1_Radio0_14ft_0.pth -> 14
        distance_pattern = re.compile(r'_(\d+)ft_')
        
        # Build mapping similar to ActivationDataset
        available_activations = {os.path.basename(f) for f in os.listdir(self.activation_dir) if f.endswith('.pth')}
        
        for mat_path in self.file_list:
            base_name = os.path.basename(mat_path).replace('.mat', '.pth')
            
            # Check for direct mapping first (regular datasets)
            if base_name in available_activations:
                pth_file_path = os.path.join(self.activation_dir, base_name)
                # Extract distance directly from the .pth filename (much faster than loading the file)
                match = distance_pattern.search(base_name)
                if match:
                    distance = int(match.group(1))
                    self.unique_distances.add(distance)
                    self.activation_map[os.path.basename(mat_path)] = pth_file_path
                    self.distances.append(distance)
                else:
                    print(f"Warning: Could not extract distance from filename: {base_name}")
            else:
                # Check for slice-based files (RF Fixed datasets)
                slice_files_found = False
                for slice_idx in range(3):
                    slice_name = base_name.replace('.pth', f'_slice{slice_idx}.pth')
                    if slice_name in available_activations:
                        slice_path = os.path.join(self.activation_dir, slice_name)
                        # Extract distance directly from slice filename
                        match = distance_pattern.search(slice_name)
                        if match:
                            distance = int(match.group(1))
                            self.unique_distances.add(distance)
                            slice_mat_name = os.path.basename(mat_path).replace('.mat', f'_slice{slice_idx}.mat')
                            self.activation_map[slice_mat_name] = slice_path
                            self.distances.append(distance)
                            slice_files_found = True
                            break
                
                if not slice_files_found:
                    print(f"Warning: No activation file found for {base_name}")
        
        # Create distance to class index mapping
        self.unique_distances = sorted(list(self.unique_distances))
        self.distance_to_class = {dist: idx for idx, dist in enumerate(self.unique_distances)}
        self.num_classes = len(self.unique_distances)
        
        print(f"Initialized dataset with {len(self.activation_map)} samples.")
        print(f"Found {self.num_classes} unique distance classes: {self.unique_distances}")
        
        if len(self.activation_map) == 0:
            raise ValueError(f"No valid activation files found in {self.activation_dir}")
    
    def __len__(self):
        return len(self.activation_map)
    
    def __getitem__(self, index):
        """
        Returns a tuple: (activation, distance_class, distance_value, filename)
        """
        # Get the filename and corresponding activation path
        filenames = list(self.activation_map.keys())
        filename = filenames[index]
        pth_file_path = self.activation_map[filename]
        
        # Load the activation data
        data = torch.load(pth_file_path, map_location='cpu')
        activation = data['activation']  # Shape: (1, 2, latent_dim) or (2, latent_dim)
        
        # Get distance from the stored distances list
        distance_value = self.distances[index]
        distance_class = self.distance_to_class[distance_value]
        
        return activation, distance_class, distance_value, filename


def train_epoch(model, train_loader, criterion, optimizer, device, debug=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        activations, labels, distances, filenames = batch
        
        # Debug: Print shapes for first batch
        if debug and batch_idx == 0:
            print(f"\n=== DEBUG: Training Batch {batch_idx} ===")
            print(f"  Raw activations shape: {activations.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Labels values (first 5): {labels[:5]}")
            print(f"  Distances (first 5): {distances[:5]}")
            print(f"  Filenames (first 3): {[f[:50] for f in filenames[:3]]}")
        
        # Handle activation shape: could be (B, 1, 2, latent_dim) or (B, 2, latent_dim)
        activations = activations.to(device)
        if debug and batch_idx == 0:
            print(f"  Activations after .to(device): {activations.shape}")
        
        if activations.dim() == 4:  # (B, 1, 2, latent_dim)
            activations = activations.squeeze(1)  # -> (B, 2, latent_dim)
            if debug and batch_idx == 0:
                print(f"  After squeeze(1) (dim==4): {activations.shape}")
        elif activations.dim() == 3:  # (B, 2, latent_dim)
            if debug and batch_idx == 0:
                print(f"  Already correct shape (dim==3): {activations.shape}")
            pass  # Already correct shape
        else:
            # Flatten and reshape if needed
            batch_size = activations.size(0)
            activations = activations.view(batch_size, -1)
            latent_dim = activations.size(1) // 2
            activations = activations.view(batch_size, 2, latent_dim)
            if debug and batch_idx == 0:
                print(f"  After reshape (else case): {activations.shape}")
        
        labels = labels.to(device).long()
        
        if debug and batch_idx == 0:
            print(f"  Final activations shape (input to model): {activations.shape}")
            print(f"  Final labels shape: {labels.shape}")
            print(f"  Model input_dim: {model.input_dim}")
            print(f"  Model num_classes: {model.classifier[-1].out_features}")
        
        optimizer.zero_grad()
        outputs = model(activations)
        
        if debug and batch_idx == 0:
            print(f"  Model outputs shape: {outputs.shape}")
            print(f"  Model outputs dtype: {outputs.dtype}")
            print(f"  Outputs (first 3 samples, first 5 classes): {outputs[:3, :5]}")
        
        loss = criterion(outputs, labels)
        
        if debug and batch_idx == 0:
            print(f"  Loss value: {loss.item()}")
            print(f"  Loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
        
        loss.backward()
        
        if debug and batch_idx == 0:
            # Print gradient norms
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms.append((name, param.grad.norm().item()))
            print(f"  Gradient norms (first 5): {grad_norms[:5]}")
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if debug and batch_idx == 0:
            print(f"  Predicted (first 10): {predicted[:10]}")
            print(f"  Labels (first 10): {labels[:10]}")
            print(f"  Correct in this batch: {(predicted == labels).sum().item()}/{labels.size(0)}")
            print("=== END DEBUG ===\n")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def get_empirical_fim(head, data_loader, device, latent_dim):
    """Calculates the empirical Fisher Information Matrix for the distance classification head."""
    fim = torch.zeros((latent_dim, latent_dim), device=device)
    num_samples = 0
    head.to(device).eval()
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in tqdm(data_loader, desc="Calculating FIM", leave=False):
        activations, labels, _, _ = batch
        
        # Handle activation shape
        activations = activations.to(device)
        if activations.dim() == 4:
            activations = activations.squeeze(1)
        elif activations.dim() == 2:
            batch_size = activations.size(0)
            activations = activations.view(batch_size, -1)
            latent_dim_flat = activations.size(1) // 2
            activations = activations.view(batch_size, 2, latent_dim_flat)
        
        # Flatten for FIM calculation
        activations_flat = activations.view(activations.size(0), -1)
        activations_flat.requires_grad = True
        
        head.zero_grad()
        
        # Reshape for head input
        latent_dim_head = activations_flat.size(1) // 2
        activations_for_head = activations_flat.view(activations_flat.size(0), 2, latent_dim_head)
        
        y_target = labels.to(device).long()
        y_hat = head(activations_for_head)
        
        loss = criterion(y_hat, y_target)
        loss.backward()
        
        if activations_flat.grad is not None:
            J = activations_flat.grad
            fim += J.T @ J
            num_samples += activations_flat.size(0)
    
    if num_samples == 0:
        return fim
    return fim / num_samples


def evaluate(model, data_loader, criterion, device, debug=False, noise_level=0.0, L=None, V=None, lambda_factor=1e-5):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating", leave=False)):
            activations, labels, distances, filenames = batch
            
            # Debug: Print shapes for first batch
            if debug and batch_idx == 0:
                print(f"\n=== DEBUG: Evaluation Batch {batch_idx} ===")
                print(f"  Raw activations shape: {activations.shape}")
                print(f"  Labels shape: {labels.shape}")
            
            # Handle activation shape: could be (B, 1, 2, latent_dim) or (B, 2, latent_dim)
            activations = activations.to(device)
            if activations.dim() == 4:  # (B, 1, 2, latent_dim)
                activations = activations.squeeze(1)  # -> (B, 2, latent_dim)
            elif activations.dim() == 3:  # (B, 2, latent_dim)
                pass  # Already correct shape
            else:
                # Flatten and reshape if needed
                batch_size = activations.size(0)
                activations = activations.view(batch_size, -1)
                latent_dim = activations.size(1) // 2
                activations = activations.view(batch_size, 2, latent_dim)
            
            # Flatten for noise injection
            activations_flat = activations.view(activations.size(0), -1)
            
            # Inject non-isotropic noise if noise_level > 0
            if noise_level > 0 and L is not None and V is not None:
                activations_flat = inject_nonisotropic_noise(
                    activations_flat, noise_level, L, V, lambda_factor=lambda_factor
                )
                # Reshape back for model input
                latent_dim = activations_flat.size(1) // 2
                activations = activations_flat.view(activations_flat.size(0), 2, latent_dim)
            
            labels = labels.to(device).long()
            
            if debug and batch_idx == 0:
                print(f"  Final activations shape (input to model): {activations.shape}")
                print(f"  Final labels shape: {labels.shape}")
            
            outputs = model(activations)
            loss = criterion(outputs, labels)
            
            if debug and batch_idx == 0:
                print(f"  Model outputs shape: {outputs.shape}")
                print(f"  Loss value: {loss.item()}")
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if debug and batch_idx == 0:
                print(f"  Predicted (first 10): {predicted[:10]}")
                print(f"  Labels (first 10): {labels[:10]}")
                print("=== END DEBUG ===\n")
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1, precision, recall, all_preds, all_labels


def main(args):
    """Main training function."""
    
    # Setup
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load partition dictionary
    pkl_file_path = os.environ.get('PKL_FILE_PATH')
    if not pkl_file_path:
        raise ValueError("PKL_FILE_PATH environment variable not set.")
    
    partition_file = os.path.join(pkl_file_path, 'rf_partition_dict_0.5.pkl')
    with open(partition_file, 'rb') as f:
        partitions = pickle.load(f)
    
    train_files = partitions['train']
    val_files = partitions['val']
    test_files = partitions['test']
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DistanceDataset(
        activation_dir=args.activation_dir,
        file_list=train_files,
        test_mode=False
    )
    val_dataset = DistanceDataset(
        activation_dir=args.activation_dir,
        file_list=val_files,
        test_mode=True
    )
    test_dataset = DistanceDataset(
        activation_dir=args.activation_dir,
        file_list=test_files,
        test_mode=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of distance classes: {train_dataset.num_classes}")
    print(f"Distance classes: {train_dataset.unique_distances}")
    
    # Debug: Print a sample from dataset
    if args.debug:
        print("\n=== DEBUG: Dataset Sample ===")
        sample_activation, sample_label, sample_distance, sample_filename = train_dataset[0]
        print(f"  Sample activation shape: {sample_activation.shape}")
        print(f"  Sample activation dtype: {sample_activation.dtype}")
        print(f"  Sample label (class): {sample_label}")
        print(f"  Sample distance (value): {sample_distance}")
        print(f"  Sample filename: {sample_filename[:80]}")
        print("=== END DEBUG ===\n")
    
    # Check if all datasets have the same classes
    if train_dataset.unique_distances != val_dataset.unique_distances:
        print("Warning: Train and val datasets have different distance classes!")
    if train_dataset.unique_distances != test_dataset.unique_distances:
        print("Warning: Train and test datasets have different distance classes!")
    
    # Use the number of classes from training set
    num_classes = train_dataset.num_classes
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Get latent dimension from a sample
    sample_activation, _, _, _ = train_dataset[0]
    if sample_activation.dim() == 4:  # (1, 1, 2, latent_dim) - unlikely but handle it
        latent_dim = sample_activation.size(3)
        sample_activation = sample_activation.squeeze(0).squeeze(0)
    elif sample_activation.dim() == 3:  # (1, 2, latent_dim)
        latent_dim = sample_activation.size(2)
        sample_activation = sample_activation.squeeze(0)
    elif sample_activation.dim() == 2:  # (2, latent_dim)
        latent_dim = sample_activation.size(1)
    else:
        # Flatten and infer
        flat_size = sample_activation.view(-1).size(0)
        latent_dim = flat_size // 2
    
    input_dim = 2 * latent_dim  # RFClassificationHead expects flattened input
    
    print(f"Detected latent dimension: {latent_dim}")
    print(f"Input dimension for head: {input_dim}")
    
    # Create model
    model = RFClassificationHead(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=args.patience // 2,
            verbose=True
        )
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            **vars(args),
            'num_classes': num_classes,
            'latent_dim': latent_dim,
            'input_dim': input_dim,
            'distance_classes': train_dataset.unique_distances
        },
        name=args.wandb_name if args.wandb_name else f"distance_classification_{args.activation_dir.split('/')[-1]}"
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            debug=args.debug and epoch == 0
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_precision, val_recall, _, _ = evaluate(
            model, val_loader, criterion, device,
            debug=args.debug and epoch == 0
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # Log to wandb
        log_dict = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'val/f1': val_f1,
            'val/precision': val_precision,
            'val/recall': val_recall,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        wandb.log(log_dict)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'distance_classes': train_dataset.unique_distances
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print("Saved best model.")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break
    
    # Load best model
    print("\nLoading best model...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calculate FIM for non-isotropic noise
    print("\nCalculating FIM for non-isotropic noise...")
    from torch.utils.data import Subset
    
    # Use a subset of training data for FIM calculation
    fim_subset_size = min(args.fim_samples, len(train_dataset))
    fim_subset_indices = np.random.choice(len(train_dataset), size=fim_subset_size, replace=False)
    fim_dataset = Subset(train_dataset, fim_subset_indices)
    fim_loader = DataLoader(fim_dataset, batch_size=args.batch_size, shuffle=False)
    
    fim = get_empirical_fim(model, fim_loader, device, input_dim)
    
    # Normalize the FIM
    trace_fim = torch.trace(fim)
    if trace_fim > 1e-10:
        fim = fim / trace_fim
        print(f"  Final FIM normalized by its trace: {trace_fim:.3e}")
    else:
        print("  Warning: Final FIM has a zero or near-zero trace. Skipping normalization.")
    
    print("Performing eigendecomposition of the final FIM...")
    L_e, V = torch.linalg.eigh(fim)
    L = torch.relu(L_e)  # Ensure non-negative eigenvalues
    print("FIM calculation and eigendecomposition complete.")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    # Test with different noise levels
    noise_levels = [0.0, 5.0, 10.0, 15.0, 20.0]  # Baseline + 4 noise levels
    all_results = {}
    baseline_preds = None
    baseline_labels = None
    
    for noise_level in noise_levels:
        print(f"\n{'='*60}")
        print(f"Evaluating with noise level: {noise_level}")
        print(f"{'='*60}")
        
        test_loss, test_acc, test_f1, test_precision, test_recall, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device,
            noise_level=noise_level,
            L=L if noise_level > 0 else None,
            V=V if noise_level > 0 else None,
            lambda_factor=args.lambda_factor
        )
        
        print(f"\nTest Results (noise_level={noise_level}):")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        
        # Classification report
        class_names = [str(dist) for dist in train_dataset.unique_distances]
        report = classification_report(
            test_labels,
            test_preds,
            target_names=class_names,
            output_dict=True
        )
        
        # Compute confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        
        # Save confusion matrix as numpy array
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_noise_{noise_level}.npy')
        np.save(cm_path, cm)
        
        # Plot and save confusion matrix as PNG
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - Distance Classification (Noise Level: {noise_level})', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Distance (ft)', fontsize=12)
        plt.ylabel('True Distance (ft)', fontsize=12)
        plt.tight_layout()
        cm_plot_path = os.path.join(args.output_dir, f'confusion_matrix_noise_{noise_level}.png')
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store predictions/labels for baseline
        if noise_level == 0.0:
            baseline_preds = test_preds
            baseline_labels = test_labels
        
        # Store results
        all_results[f'noise_{noise_level}'] = {
            'noise_level': float(noise_level),
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'classification_report': convert_to_native(report),
            'confusion_matrix_path': cm_path,
            'confusion_matrix_plot_path': cm_plot_path
        }
        
        # Log to wandb
        wandb.log({
            f'test_noise_{noise_level}/loss': test_loss,
            f'test_noise_{noise_level}/accuracy': test_acc,
            f'test_noise_{noise_level}/f1': test_f1,
            f'test_noise_{noise_level}/precision': test_precision,
            f'test_noise_{noise_level}/recall': test_recall,
            f'confusion_matrix_noise_{noise_level}': wandb.Image(cm_plot_path)
        })
    
    # Use baseline results for the main test_results (for backward compatibility)
    baseline_results = all_results['noise_0.0']
    test_loss = baseline_results['test_loss']
    test_acc = baseline_results['test_accuracy']
    test_f1 = baseline_results['test_f1']
    test_precision = baseline_results['test_precision']
    test_recall = baseline_results['test_recall']
    
    # Use stored baseline predictions and labels
    test_preds = baseline_preds
    test_labels = baseline_labels
    
    # Print baseline results summary
    print(f"\n{'='*60}")
    print("Baseline Results (Noise Level: 0.0):")
    print(f"{'='*60}")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    
    # Classification report for baseline
    class_names = [str(dist) for dist in train_dataset.unique_distances]
    report = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        output_dict=True
    )
    
    print("\nClassification Report (Baseline):")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Compute confusion matrix for baseline
    cm = confusion_matrix(test_labels, test_preds)
    
    # Save baseline confusion matrix as numpy array (for backward compatibility)
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.npy')
    np.save(cm_path, cm)
    print(f"Baseline confusion matrix saved as numpy array to {cm_path}")
    
    # Plot and save baseline confusion matrix as PNG (for backward compatibility)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Distance Classification (Baseline)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Distance (ft)', fontsize=12)
    plt.ylabel('True Distance (ft)', fontsize=12)
    plt.tight_layout()
    cm_plot_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Baseline confusion matrix plot saved to {cm_plot_path}")
    
    # Save test results (include all noise levels)
    test_results = {
        'baseline': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'num_classes': num_classes,
            'distance_classes': [int(d) for d in train_dataset.unique_distances],
            'classification_report': convert_to_native(report),
            'predictions': [int(p) for p in (test_preds.tolist() if isinstance(test_preds, np.ndarray) else test_preds)],
            'labels': [int(l) for l in (test_labels.tolist() if isinstance(test_labels, np.ndarray) else test_labels)]
        },
        'all_noise_levels': convert_to_native(all_results)
    }
    
    # Convert all numpy types in the dict
    test_results = convert_to_native(test_results)
    
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"\nTest results (all noise levels) saved to {results_path}")
    
    # Log final baseline metrics to wandb
    wandb.log({
        'test/loss': test_loss,
        'test/accuracy': test_acc,
        'test/f1': test_f1,
        'test/precision': test_precision,
        'test/recall': test_recall,
        'confusion_matrix': wandb.Image(cm_plot_path)
    })
    
    # Print summary table
    print(f"\n{'='*60}")
    print("Summary of Results Across All Noise Levels:")
    print(f"{'='*60}")
    print(f"{'Noise Level':<15} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    for noise_level in noise_levels:
        res = all_results[f'noise_{noise_level}']
        print(f"{noise_level:<15.1f} {res['test_accuracy']:<12.4f} {res['test_f1']:<12.4f} "
              f"{res['test_precision']:<12.4f} {res['test_recall']:<12.4f}")
    print("="*60)
    
    wandb.finish()
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train distance classification from activations')
    
    # Paths
    parser.add_argument('--activation_dir', type=str, required=True, 
                       help='Path to directory containing activation .pth files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results and model checkpoints')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for classification head')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--patience', type=int, default=8, help='Patience for early stopping')
    parser.add_argument('--use_lr_scheduler', action='store_true',
                       help='Use learning rate scheduler (ReduceLROnPlateau)')
    
    # System
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loader workers')
    
    # Wandb
    parser.add_argument('--wandb_project', type=str, default='distance-classification',
                       help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='WandB run name (optional)')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information (shapes, values) for first batch')
    
    # FIM and noise parameters
    parser.add_argument('--fim_samples', type=int, default=1000,
                       help='Number of samples for FIM calculation')
    parser.add_argument('--lambda_factor', type=float, default=0.01,
                       help='Lambda factor for FIM regularization in non-isotropic noise')
    
    args = parser.parse_args()
    main(args)

