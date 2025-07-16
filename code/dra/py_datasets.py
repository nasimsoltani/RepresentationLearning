
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import json
import argparse
import numpy as np
import re
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Adjust sys.path to allow imports from the 'rep_lr' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rep_lr.models import (ComplexSequenceProjector, Encoder, 
                   RFClassificationHead, ChannelEstimationHead, CFOEstimationHead)


class ActivationDataset(Dataset):
    """
    A PyTorch Dataset to load saved activations. It can optionally add structured
    (anisotropic) or unstructured (isotropic) noise to the activations for
    privacy-preserving representation learning experiments.
    """
    def __init__(self, activations_dir, args):
        """
        Args:
            activations_dir (str): Directory where the .pth activation files are saved.
            args (Namespace): Arguments containing model_path, noise_type, noise_level, etc.
        """
        self.activations_dir = activations_dir
        self.activation_files = [os.path.join(self.activations_dir, f) for f in os.listdir(self.activations_dir) if f.endswith('.pth')]
        
        if not self.activation_files:
            print("Warning: No .pth files found in the activations directory.")
        else:
            print(f"Found {len(self.activation_files)} activation files.")

        self.noise_type = getattr(args, 'noise_type', 'none')
        self.noise_level = getattr(args, 'noise_level', 0.0)

        if self.noise_type in ['anisotropic', 'isotropic']:
            print(f"Initializing dataset with {self.noise_type} noise (level: {self.noise_level})")
            # This flag prevents recursive initialization when creating a temporary dataset for FIM
            if not getattr(args, '_is_fim_computation', False):
                self._initialize_fim_and_noise_components(args)
        else:
            print("Initializing dataset without noise.")

    def _initialize_fim_and_noise_components(self, args):
        """Loads the model, calculates the FIM, and prepares for noise generation."""
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Load model heads and corresponding training arguments
        model, train_args = self._load_model_heads_from_checkpoint(args.model_path)
        self.model = model.to(self.device)
        self.model.eval()
        self.latent_dim = 2 * train_args.d2

        # Use a subset of data to compute FIM
        num_fim_samples = min(500, int(len(self.activation_files) * 0.2))
        if num_fim_samples == 0 and len(self.activation_files) > 0:
            num_fim_samples = len(self.activation_files)
        
        if num_fim_samples == 0:
            raise ValueError("Not enough activation files to compute FIM. Need at least 1.")

        print(f"Using {num_fim_samples} samples to compute FIM.")
        fim_files = np.random.choice(self.activation_files, num_fim_samples, replace=False).tolist()

        # Calculate total FIM across all tasks
        tasks = train_args.task if isinstance(train_args.task, list) else [train_args.task]
        F_total = torch.zeros((self.latent_dim, self.latent_dim), device=self.device)

        for task_name in tasks:
            print(f"  Calculating FIM for task: {task_name}")
            fim = self._get_empirical_fim_for_task(task_name, fim_files)
            F_total += fim

        # Eigendecomposition of the total FIM
        print("Performing eigendecomposition on total FIM...")
        L, V = torch.linalg.eigh(F_total)
        self.L = torch.relu(L)  # Eigenvalues
        self.V = V              # Eigenvectors

    def _load_model_heads_from_checkpoint(self, model_path):
        """Loads only the task-specific heads from a model checkpoint."""
        model_dir = os.path.dirname(model_path)
        args_path = os.path.join(model_dir, 'args.json')
        if not os.path.exists(args_path):
            raise FileNotFoundError(f"args.json not found in {model_dir}")
        with open(args_path, 'r') as f:
            train_args = argparse.Namespace(**json.load(f))

        is_mtl = getattr(train_args, 'mtl', False)
        heads = torch.nn.ModuleDict()
        ID_class_dict = {f'Radio{i}': i for i in range(16)}
        num_classes = len(ID_class_dict)
        tasks = train_args.task if is_mtl else [train_args.task]

        for task in tasks:
            if task == 'rf_fingerprinting':
                heads[task] = RFClassificationHead(input_dim=2*train_args.d2, num_classes=num_classes, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)
            elif task == 'channel_estimation':
                heads[task] = ChannelEstimationHead(input_dim=2*train_args.d2, hidden_dim=train_args.head_hidden_dim, output_length=52, dropout=train_args.dropout)
            elif task == 'cfo_estimation':
                heads[task] = CFOEstimationHead(input_dim=2*train_args.d2, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)
        
        model = torch.nn.ModuleDict({'heads': heads})
        checkpoint = torch.load(model_path, map_location='cpu')

        if is_mtl:
            model['heads'].load_state_dict(checkpoint['heads_state_dict'])
        else:
            task_head_state_dict = {}
            if 'module_2' in checkpoint:
                task_head_state_dict = checkpoint['module_2']
            elif 'model_state_dict' in checkpoint:
                task_head_state_dict = {k.replace('2.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('2.')}
            else:
                raise KeyError("Could not find head weights in the checkpoint.")
            model['heads'][tasks[0]].load_state_dict(task_head_state_dict)
        
        return model, train_args

    def _get_empirical_fim_for_task(self, task_name, fim_files):
        """Calculates the FIM for a single task using the gradients of the loss w.r.t. the activation."""
        fim = torch.zeros((self.latent_dim, self.latent_dim), device=self.device)
        num_samples = 0
        task_head = self.model['heads'][task_name]
        criterion = nn.CrossEntropyLoss() if task_name == 'rf_fingerprinting' else nn.MSELoss()

        for file_path in fim_files:
            data = torch.load(file_path, map_location=self.device)
            # Load activation and ensure it has proper batch dimension
            z = data['activation'].squeeze(0)  # Remove any extra dimensions: (1, 2, 256) -> (2, 256)
            z = z.view(1, -1)  # Flatten to (1, 512) for task head input
            z.requires_grad_(True)

            try:
                y_target = self._get_target_for_task(data, task_name)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping file {os.path.basename(file_path)} for FIM calc. Reason: {e}")
                continue

            task_head.zero_grad()
            y_hat = task_head(z)
            
            if task_name == 'rf_fingerprinting': 
                y_target = y_target.long()
            elif task_name == 'cfo_estimation': 
                # CFO head outputs (1, 1), so target should also be (1, 1)
                if y_target.dim() == 0: y_target = y_target.unsqueeze(0).unsqueeze(0)  # scalar -> (1, 1)
                elif y_target.dim() == 1: y_target = y_target.unsqueeze(0)  # (1,) -> (1, 1)
            elif task_name == 'channel_estimation':
                # Channel head outputs (1, 2, 52), so target should also have batch dimension
                if y_target.dim() == 2: y_target = y_target.unsqueeze(0)  # (2, 52) -> (1, 2, 52)

            loss = criterion(y_hat, y_target)
            loss.backward()

            if z.grad is not None:
                J = z.grad.view(-1)  # Flatten gradient to 1D: (1, 512) -> (512,)
                fim += torch.outer(J, J)  # Outer product: (512,) x (512,) -> (512, 512)
                num_samples += z.size(0)
        
        return fim / num_samples if num_samples > 0 else fim
    
    def _get_target_for_task(self, data, task_name):
        """
        Retrieves the ground truth label for a given task from the loaded data.
        NOTE: This assumes that labels ('rf_label', 'cfo_label') are saved in the .pth files.
        """
        if task_name == 'rf_fingerprinting':
            if 'rf_label' not in data: raise KeyError("'rf_label' not found in data file.")
            # Keep batch dimension for CrossEntropyLoss: [1] -> [1]
            label = data['rf_label'].squeeze()
            if label.dim() == 0: label = label.unsqueeze(0)  # [] -> [1]
            return label
        elif task_name == 'cfo_estimation':
            if 'cfo_label' not in data: raise KeyError("'cfo_label' not found in data file.")
            # CFO label: convert to float32 and ensure proper shape for MSELoss
            label = data['cfo_label'].squeeze().float()  # Convert to float32
            return label
        elif task_name == 'channel_estimation':
            if 'channel_label' not in data: raise KeyError("'channel_label' not found in data file.")
            # Channel label: use the actual channel labels, not input
            return data['channel_label'].squeeze(0)
        else:
            raise ValueError(f"Unknown task for target retrieval: {task_name}")

    def _get_anisotropic_noise(self, z):
        """Generates anisotropic noise based on the FIM's eigenspectrum."""
        if self.noise_level == 0.0: return torch.zeros_like(z)
        
        # Store original shape and flatten z for noise generation
        original_shape = z.shape
        z_flat = z.view(-1)  # Flatten to match FIM dimensions
        
        aniso_variances = self.noise_level / (self.L.to(z.device) + 1e-6)
        scale_tril = torch.diag(torch.sqrt(aniso_variances))
        noise_flat = (scale_tril @ torch.randn_like(z_flat).unsqueeze(-1)).squeeze(-1)
        
        # Reshape back to original activation shape
        return noise_flat.view(original_shape)

    def _get_isotropic_noise(self, z):
        """Generates isotropic noise with the same total power as the anisotropic noise."""
        if self.noise_level == 0.0: return torch.zeros_like(z)

        total_power = torch.sum(self.noise_level / (self.L + 1e-6))
        iso_variance = total_power / self.latent_dim
        return torch.randn_like(z) * torch.sqrt(iso_variance)

    def __len__(self):
        return len(self.activation_files)

    def __getitem__(self, index):
        """
        Returns a dictionary containing the (potentially noisy) activation and original data.
        """
        activation_path = self.activation_files[index]
        data = torch.load(activation_path)
        
        activation = data['activation'].squeeze(0)
        
        # Add noise if configured
        if self.noise_type == 'anisotropic':
            activation += self._get_anisotropic_noise(activation.to(self.device)).cpu()
        elif self.noise_type == 'isotropic':
            activation += self._get_isotropic_noise(activation.to(self.device)).cpu()

        return {
            'activation': activation,
            'RF_X': data['RF_X'].squeeze(0),
            'CFO_X': data['CFO_X'].squeeze(0),
            'Channel_X': data['Channel_X'].squeeze(0),
            'filename': data['filename'],
            'rf_label': data.get('rf_label', torch.tensor(-1)).squeeze(),
            'cfo_label': data.get('cfo_label', torch.tensor(-1.0)).squeeze()
        } 