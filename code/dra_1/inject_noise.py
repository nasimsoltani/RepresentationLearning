import torch

def inject_isotropic_noise(z, noise_level):
    """
    Injects isotropic Gaussian noise with a specified total variance.

    Args:
        z (torch.Tensor): The input tensor (N, D_z).
        noise_level (float): The total variance of the noise (σ_total^2).

    Returns:
        torch.Tensor: The tensor with added noise.
    """
    if noise_level > 0:
        # Normalize activations to have unit norm, but keep track of original norm
        z_norm = torch.linalg.norm(z, dim=1, keepdim=True)
        z_normalized = z / (z_norm + 1e-7) # Add epsilon for stability

        latent_dim = z.size(-1)
        # Distribute total variance equally across all dimensions
        per_dim_variance = noise_level / latent_dim
        # Convert variance to standard deviation for noise generation
        per_dim_stddev = torch.sqrt(torch.tensor(per_dim_variance, device=z.device))
        
        epsilon = torch.randn_like(z_normalized) * per_dim_stddev

        #for debugging print variance of epsilon
        #print("variance of epsilon", torch.var(epsilon, dim=0))

        # Add noise to the normalized vector and then scale it back to its original magnitude
        z_noisy_normalized = z_normalized + epsilon
        # Re-normalize to preserve the norm before scaling back
        #z_noisy_renormalized = z_noisy_normalized / (torch.linalg.norm(z_noisy_normalized, dim=1, keepdim=True) + 1e-7)
        z_noisy = z_noisy_normalized * z_norm
        return z_noisy
    return z


def inject_nonisotropic_noise(z, noise_level, L, V, lambda_factor=1e-5):
    """
    Injects anisotropic Gaussian noise based on FIM eigendecomposition.

    Args:
        z (torch.Tensor): The latent representation (N, D_z).
        noise_level (float): The total noise variance budget (σ_total^2), relative to unit-norm activations.
        L (torch.Tensor): Eigenvalues of the FIM (D_z,).
        V (torch.Tensor): Eigenvectors of the FIM (D_z, D_z).

    Returns:
        torch.Tensor: The tensor with added anisotropic noise.
    """
    if noise_level > 0 and L is not None and V is not None:
        # Normalize activations to have unit norm, but keep track of original norm
        z_norm = torch.linalg.norm(z, dim=1, keepdim=True)
        z_normalized = z / (z_norm + 1e-7)

        # The total noise variance is distributed inversely proportional to eigenvalues.
        # Variances σ_i^2 = c / λ_i, where sum(σ_i^2) = noise_level.
        # This implies c = noise_level / sum(1 / λ_i).
        lambda_ = torch.sum(L)*lambda_factor
        inv_L = 1.0 / (L + lambda_)  # Add small epsilon for stability
        c = noise_level / torch.sum(inv_L)
        aniso_variances = c * inv_L

        # Construct the covariance matrix in the standard basis
        # Sigma_noise = V * diag(aniso_variances) * V^T
        aniso_cov = V @ torch.diag(aniso_variances) @ V.T

        # Generate noise with this covariance structure
        try:
            # More stable way to generate correlated noise
            scale_tril = torch.linalg.cholesky(aniso_cov)
            base_noise = torch.randn_like(z_normalized) # (N, D_z)
            aniso_noise = (scale_tril @ base_noise.T).T
            #for debugging print variance of aniso_noise
            #print("variance of aniso_noise", torch.var(aniso_noise, dim=0))
            
            # Add noise to normalized vector and scale back
            z_noisy_normalized = z_normalized + aniso_noise
            # Re-normalize to preserve the norm before scaling back
            #z_noisy_renormalized = z_noisy_normalized / (torch.linalg.norm(z_noisy_normalized, dim=1, keepdim=True) + 1e-7)
            z_noisy = z_noisy_normalized * z_norm
            return z_noisy
        except torch.linalg.LinAlgError:
            # Fallback for numerical instability
            print("Warning: Cholesky decomposition failed. Using V @ Omega_sqrt method.")
            Omega_sqrt = torch.diag(torch.sqrt(aniso_variances))
            base_noise = torch.randn_like(z_normalized) # (N, D_z)
            aniso_noise = (V @ Omega_sqrt @ base_noise.T).T

            # Add noise to normalized vector and scale back
            z_noisy_normalized = z_normalized + aniso_noise
            # Re-normalize to preserve the norm before scaling back
            # z_noisy_renormalized = z_noisy_normalized / (torch.linalg.norm(z_noisy_normalized, dim=1, keepdim=True) + 1e-7)
            z_noisy = z_noisy_normalized * z_norm
            return z_noisy
    return z 