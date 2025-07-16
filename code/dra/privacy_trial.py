import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Synthetic Data Generation (Updated for more distinct tasks) ---
def create_synthetic_data(num_samples=1000, input_dim=50):
    """
    Creates a dataset where tasks depend on very distinct, low-dimensional subspaces of X.
    This creates a more "spiky" FIM, ideal for showcasing the anisotropic method.
    """
    X = torch.randn(num_samples, input_dim)

    # Task 1 depends only on the first 3 features
    W1 = torch.randn(3, 1)
    Y1 = X[:, :3] @ W1 + 0.1 * torch.randn(num_samples, 1)

    # Task 2 depends only on features 20 to 22 (a completely different subspace)
    W2 = torch.randn(3, 1)
    Y2 = X[:, 20:23] @ W2 + 0.1 * torch.randn(num_samples, 1)

    return X, Y1, Y2

# --- 2. Model Definitions ---
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class TaskHead(nn.Module):
    def __init__(self, latent_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, output_dim))
    def forward(self, z):
        return self.net(z)

class MTLModel(nn.Module):
    def __init__(self, encoder, head1, head2):
        super().__init__()
        self.encoder = encoder
        self.head1 = head1
        self.head2 = head2
    def forward(self, x):
        z = self.encoder(x)
        return self.head1(z), self.head2(z)

class Adversary(nn.Module):
    """Tries to reconstruct X from Z'."""
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, z_prime):
        return self.net(z_prime)

# --- 3. FIM Calculation (Corrected Function) ---
def get_empirical_fim(model, data_loader, task_idx, latent_dim):
    """Calculates the empirical Fisher Information Matrix for a given task."""
    fim = torch.zeros((latent_dim, latent_dim))
    criterion = nn.MSELoss()
    num_samples = 0

    model.eval()

    for x_batch, y1_batch, y2_batch in data_loader:
        model.zero_grad()

        # 1. Compute Z, the non-leaf tensor
        z = model.encoder(x_batch)

        # 2. Explicitly tell PyTorch to save the gradient for this intermediate tensor
        z.retain_grad()

        # 3. Get the correct head and target for the specified task
        task_head = model.head1 if task_idx == 0 else model.head2
        y_target = y1_batch if task_idx == 0 else y2_batch

        # 4. Compute loss ONLY from Z onwards to get the correct gradient
        y_hat = task_head(z)
        loss = criterion(y_hat, y_target)

        # 5. Backpropagate on this specific loss to populate z.grad
        loss.backward()

        # 6. Get the Jacobian (gradient of loss w.r.t z) and accumulate FIM
        J = z.grad
        fim += J.T @ J
        num_samples += x_batch.size(0)

    return fim / num_samples

# --- Main Experiment Logic ---

# Hyperparameters
INPUT_DIM = 50
LATENT_DIM = 10
EPOCHS_MTL = 20
EPOCHS_ADV = 15
LR = 1e-3
BATCH_SIZE = 64
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]

# Data
X, Y1, Y2 = create_synthetic_data(num_samples=2000, input_dim=INPUT_DIM)
# Split data: 1000 for MTL, 500 for FIM/adv_train, 500 for test
train_dataset = TensorDataset(X[:1000], Y1[:1000], Y2[:1000])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

adv_dataset = TensorDataset(X[1000:1500], Y1[1000:1500], Y2[1000:1500])
adv_loader = DataLoader(adv_dataset, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(X[1500:], Y1[1500:], Y2[1500:])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Initialize and Train MTL Model
encoder = Encoder(INPUT_DIM, LATENT_DIM)
head1 = TaskHead(LATENT_DIM)
head2 = TaskHead(LATENT_DIM)
mtl_model = MTLModel(encoder, head1, head2)
optimizer = optim.Adam(mtl_model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("## Training Multi-Task Model... ##")
for epoch in range(EPOCHS_MTL):
    for x_batch, y1_batch, y2_batch in train_loader:
        optimizer.zero_grad()
        y1_hat, y2_hat = mtl_model(x_batch)
        loss1 = criterion(y1_hat, y1_batch)
        loss2 = criterion(y2_hat, y2_batch)
        total_loss = loss1 + loss2
        total_loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS_MTL}, Loss: {total_loss.item():.4f}")

# --- 4. Get FIM and Latent Representations ---
print("\n## Calculating FIMs... ##")
fim1 = get_empirical_fim(mtl_model, adv_loader, 0, LATENT_DIM)
fim2 = get_empirical_fim(mtl_model, adv_loader, 1, LATENT_DIM)

# Combined FIM for anisotropic noise design
F_total = fim1 + fim2
L, V = torch.linalg.eigh(F_total) # Eigendecomposition
L = torch.relu(L) # Ensure eigenvalues are non-negative

# Get clean latent vectors Z for the test set
mtl_model.eval()
with torch.no_grad():
    Z_test = torch.cat([mtl_model.encoder(x) for x, _, _ in test_loader], dim=0)
    X_test = torch.cat([x for x, _, _ in test_loader], dim=0)

# --- 5. Run Privacy-Utility Experiments ---
results = {
    "iso": {"utility": [], "privacy": []},
    "aniso": {"utility": [], "privacy": [], "theory_bound": []}
}

# Calculate Sigma_Z for the theoretical bound
Z_test_mean = Z_test.mean(dim=0)
Z_centered = Z_test - Z_test_mean
Sigma_Z = (Z_centered.T @ Z_centered) / (len(Z_test) - 1)

for noise_level in NOISE_LEVELS:
    print(f"\n--- Testing Noise Level: {noise_level} ---")

    # --- Anisotropic Noise ---
    aniso_variances = noise_level / (L + 1e-6)
    Omega = torch.diag(aniso_variances)
    aniso_cov = V @ Omega @ V.T

    try:
        scale_tril = torch.linalg.cholesky(aniso_cov)
        base_noise = torch.randn_like(Z_test)
        aniso_noise = (scale_tril @ base_noise.T).T
        Z_prime_aniso = Z_test + aniso_noise
    except torch.linalg.LinAlgError:
        print("Warning: Cholesky decomposition failed. Using V@Omega_sqrt method.")
        Omega_sqrt = torch.diag(torch.sqrt(aniso_variances))
        base_noise = torch.randn_like(Z_test)
        aniso_noise = (V @ Omega_sqrt @ base_noise.T).T
        Z_prime_aniso = Z_test + aniso_noise

    # --- Isotropic Noise ---
    total_aniso_power = torch.sum(aniso_variances)
    iso_variance = total_aniso_power / LATENT_DIM
    iso_noise = torch.randn_like(Z_test) * torch.sqrt(iso_variance)
    Z_prime_iso = Z_test + iso_noise

    # --- Calculate Theoretical Bound for Anisotropic Case ---
    # The bound is [det(I + Sigma_Z * inv(Sigma_Noise))]^(-1/d_X)
    # This simplifies from the full formula because our input X is standard normal.
    if noise_level > 0:
        # Add a small epsilon for numerical stability of the inverse
        noise_cov_stable = aniso_cov + 1e-9 * torch.eye(LATENT_DIM)
        inv_noise_cov = torch.inverse(noise_cov_stable)
        det_term = torch.det(torch.eye(LATENT_DIM) + Sigma_Z @ inv_noise_cov)
        bound = torch.pow(det_term, -1.0 / INPUT_DIM)
        results["aniso"]["theory_bound"].append(bound.item())
    else:
        results["aniso"]["theory_bound"].append(0.0) # No noise, no privacy bound

    # Evaluate for both noise types
    for name, z_prime in [("aniso", Z_prime_aniso), ("iso", Z_prime_iso)]:
        # 1. Calculate Utility Loss
        y1_hat_noisy, y2_hat_noisy = head1(z_prime), head2(z_prime)
        y1_test = torch.cat([y1 for _, y1, _ in test_loader], dim=0)
        y2_test = torch.cat([y2 for _, _, y2 in test_loader], dim=0)
        utility_loss = (criterion(y1_hat_noisy, y1_test) + criterion(y2_hat_noisy, y2_test)).item()
        results[name]["utility"].append(utility_loss)

        # 2. Calculate Privacy (Train Adversary and get Reconstruction Error)
        adversary = Adversary(LATENT_DIM, INPUT_DIM)
        adv_optimizer = optim.Adam(adversary.parameters(), lr=LR)

        noisy_dataset = TensorDataset(z_prime.detach(), X_test)
        noisy_loader = DataLoader(noisy_dataset, batch_size=BATCH_SIZE)

        for _ in range(EPOCHS_ADV):
            for z_p_batch, x_batch in noisy_loader:
                adv_optimizer.zero_grad()
                x_hat = adversary(z_p_batch)
                recon_loss = criterion(x_hat, x_batch)
                recon_loss.backward()
                adv_optimizer.step()

        privacy_loss = recon_loss.item()
        results[name]["privacy"].append(privacy_loss)
        print(f"  {name.upper()} -> Utility Loss: {utility_loss:.4f}, Reconstruction Error (Privacy): {privacy_loss:.4f}")

# --- 6. Visualize FIM and Noise Covariance ---
print("\n## Visualizing FIM and Noise Covariance Matrices ##")
vis_noise_level = 0.4
vis_aniso_variances = vis_noise_level / (L + 1e-6)
vis_Omega = torch.diag(vis_aniso_variances)
vis_aniso_cov = V @ vis_Omega @ V.T

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
matrices = [fim1, fim2, F_total, vis_aniso_cov]
titles = ["FIM (Task 1)", "FIM (Task 2)", "Total FIM", f"Anisotropic Noise Cov (σ²={vis_noise_level})"]

for ax, matrix, title in zip(axes, matrices, titles):
    im = ax.imshow(matrix.detach().numpy(), cmap='viridis')
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax)

plt.suptitle("Matrix Visualizations", fontsize=18, fontweight='bold')
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()


# --- 7. Plot Privacy-Utility Trade-off ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(results["iso"]["utility"], results["iso"]["privacy"], 'o-', label="Isotropic (Naive) Noise", markersize=8, color='crimson')
ax.plot(results["aniso"]["utility"], results["aniso"]["privacy"], 's-', label="Anisotropic (FIM-based) Noise", markersize=8, color='darkgreen')
ax.set_xlabel("Utility Loss (Lower is Better)", fontsize=14)
ax.set_ylabel("Reconstruction Error (Higher is Better Privacy)", fontsize=14)
ax.set_title("Privacy-Utility Trade-off", fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.show()

# --- 8. Plot Empirical vs. Theoretical Bound ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(NOISE_LEVELS, results["aniso"]["privacy"], 's-', label="Empirical Error (Anisotropic)", markersize=8, color='darkgreen')
ax.plot(NOISE_LEVELS, results["iso"]["privacy"], 'o-', label="Empirical Error (Isotropic)", markersize=8, color='crimson', alpha=0.5)
ax.plot(NOISE_LEVELS, results["aniso"]["theory_bound"], 'd--', label="Theoretical Lower Bound (Anisotropic)", markersize=8, color='navy')

ax.set_xlabel("Noise Level (σ²)", fontsize=14)
ax.set_ylabel("Reconstruction Error", fontsize=14)
ax.set_title("Empirical Reconstruction Error vs. Theoretical Lower Bound", fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.show()
