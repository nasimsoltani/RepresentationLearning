import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import random

def inject_isotropic_noise(z, noise_level):
    """
    Injects isotropic noise into a latent vector (any shape).
    noise_level = σ_t (total noise budget).
    """
    if noise_level > 0:
        # Flatten each sample to (1, D), this we are doing in our attack and utility scripts before we pass z to this function.
        z = z.view(1, -1)
        z_norm = torch.linalg.norm(z, dim=1, keepdim=True)  # shape (1,1)
        z_normalized = z / (z_norm + 1e-9)
        latent_dim = z.size(-1)

        per_dim_variance = noise_level / latent_dim
        epsilon = torch.randn_like(z_normalized) * torch.sqrt(
            torch.tensor(per_dim_variance, device=z.device)
        )

        z_noisy_normalized = z_normalized + epsilon
        z_noisy = z_noisy_normalized * z_norm
        actual_noise = epsilon * z_norm
        return z_noisy, actual_noise
    return z, torch.zeros_like(z)


root_dir = "/scratch/10608/aadharsh_aadhithya/data/rep_lr/activations_exps/run_2/rf_cfo_channel"
if not os.path.isdir(root_dir):
    raise FileNotFoundError(f"Directory not found: {root_dir}. Please ensure the path is correct.")

activations_files = os.listdir(root_dir)
num_points = 2000
random.shuffle(activations_files)
activations_subset = activations_files[:num_points]
print(f"Found {len(activations_files)} activations, using a random subset of {len(activations_subset)} for verification.")


powers = []
noise_levels = [10, 15, 20, 25, 30] # σ_t
for noise_level in noise_levels:
    powers = []
    for activation in activations_subset:
        activation_path = os.path.join(root_dir, activation)
        activation_data = torch.load(activation_path)
        z = activation_data['activation']

        z = z.reshape(1, -1)

        z_noisy, actual_noise = inject_isotropic_noise(z, noise_level)

        noise = z_noisy - z

        # Compute total signal and noise power
        signal_power = (z ** 2).sum().item()
        noise_power = (noise ** 2).sum().item()

        powers.append(signal_power / noise_power)

    print(f"Expected SNR (1/noise_level) : {1/noise_level:.6f}")
    print(f"Empirical mean SNR          : {np.mean(powers):.6f}")
