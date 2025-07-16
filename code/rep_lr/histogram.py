import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from py_datasets import TrainDataset
from torch.utils.data import DataLoader

# Create ID class dict like in main.py
ID_class_dict = {}
for i in range(16):
    this_key = 'Radio' + str(i)
    ID_class_dict[this_key] = i
print("Class mapping (ID_class_dict):", ID_class_dict)
num_classes = len(ID_class_dict)

# Path to your dataset pickle file
pkl_dataset_path = "/scratch/10608/aadharsh_aadhithya/data/rep_lr/OracleDatasetProcessed-arranged/rf_partition_dict_0.5.pkl"

# Load the pickle data
with open(pkl_dataset_path, 'rb') as handle:
    content = pickle.load(handle)

train_list = content['train']
val_list = content['val']
max_cfo = content['max_cfo']

# Dataset arguments
dataset_args = argparse.Namespace(slice_len=1024)

# Initialize dataset and dataloader
train_dataset = TrainDataset(train_list, ID_class_dict, dataset_args, max_cfo)
train_dl = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)

print(f"Total samples in train dataset: {len(train_dataset)}")
print(f"Max CFO value (used for normalization): {max_cfo}")

# Containers to collect CFO_y values (normalized and unnormalized)
cfo_y_values_normalized = []
cfo_y_values_unnormalized = []
classwise_cfo_normalized = defaultdict(list)
classwise_cfo_unnormalized = defaultdict(list)

print("Collecting CFO_y values from all samples...")
for batch in tqdm(train_dl, desc="Processing batches", unit="batch"):
    _, RF_y, _, CFO_y, _, _, _ = batch

    RF_y_np = RF_y.cpu().numpy().flatten()
    CFO_y_normalized = CFO_y.cpu().numpy().flatten()
    CFO_y_unnormalized = CFO_y_normalized * max_cfo  # Convert back to unnormalized

    cfo_y_values_normalized.extend(CFO_y_normalized)
    cfo_y_values_unnormalized.extend(CFO_y_unnormalized)

    # Add to class-wise dictionaries
    for label, cfo_norm, cfo_unnorm in zip(RF_y_np, CFO_y_normalized, CFO_y_unnormalized):
        classwise_cfo_normalized[int(label)].append(cfo_norm)
        classwise_cfo_unnormalized[int(label)].append(cfo_unnorm)

print(f"Total collected CFO_y values: {len(cfo_y_values_normalized)}")

# =============================================
# NORMALIZED VERSION PLOTS AND STATISTICS
# =============================================

# === Normalized Histogram Plot ===
plt.figure(figsize=(12, 8))
plt.hist(cfo_y_values_normalized, bins=50, alpha=0.7, edgecolor='black')
plt.title('Histogram of CFO_y Values (Normalized by max_cfo)', fontsize=16)
plt.xlabel('CFO_y (Normalized)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)

# Statistics for normalized
mean_cfo_norm = float(np.mean(cfo_y_values_normalized))
std_cfo_norm = float(np.std(cfo_y_values_normalized))
plt.axvline(mean_cfo_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cfo_norm:.4f}')
plt.axvline(mean_cfo_norm + std_cfo_norm, color='orange', linestyle='--', alpha=0.7, label=f'+1 Std: {std_cfo_norm:.4f}')
plt.axvline(mean_cfo_norm - std_cfo_norm, color='orange', linestyle='--', alpha=0.7, label=f'-1 Std')
plt.legend()
plt.tight_layout()
plt.savefig('cfo_y_histogram_normalized.png', dpi=300, bbox_inches='tight')
plt.show()

# === Normalized Global Violin Plot ===
plt.figure(figsize=(10, 6))
sns.violinplot(data=cfo_y_values_normalized, inner="box", linewidth=1.25)
plt.title('Violin Plot of CFO_y Values (Normalized by max_cfo)', fontsize=16)
plt.ylabel('CFO_y (Normalized)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cfo_y_violinplot_normalized.png', dpi=300, bbox_inches='tight')
plt.show()

# === Normalized Class-wise Violin Plot ===
plt.figure(figsize=(14, 6))
sns.violinplot(data=[classwise_cfo_normalized[i] for i in range(num_classes)], inner="box", linewidth=1.25)
plt.title('Class-wise Violin Plot of CFO_y Values (Normalized)', fontsize=16)
plt.xlabel('Radio ID (Class)', fontsize=14)
plt.ylabel('CFO_y (Normalized)', fontsize=14)
plt.xticks(ticks=list(range(num_classes)), labels=[f'Radio{i}' for i in range(num_classes)], rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cfo_y_classwise_violinplot_normalized.png', dpi=300)
plt.show()

# =============================================
# UNNORMALIZED VERSION PLOTS AND STATISTICS
# =============================================

# === Unnormalized Histogram Plot ===
plt.figure(figsize=(12, 8))
plt.hist(cfo_y_values_unnormalized, bins=50, alpha=0.7, edgecolor='black')
plt.title('Histogram of CFO_y Values (Unnormalized)', fontsize=16)
plt.xlabel('CFO_y (Hz)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)

# Statistics for unnormalized
mean_cfo_unnorm = float(np.mean(cfo_y_values_unnormalized))
std_cfo_unnorm = float(np.std(cfo_y_values_unnormalized))
plt.axvline(mean_cfo_unnorm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cfo_unnorm:.2f} Hz')
plt.axvline(mean_cfo_unnorm + std_cfo_unnorm, color='orange', linestyle='--', alpha=0.7, label=f'+1 Std: {std_cfo_unnorm:.2f} Hz')
plt.axvline(mean_cfo_unnorm - std_cfo_unnorm, color='orange', linestyle='--', alpha=0.7, label=f'-1 Std')
plt.legend()
plt.tight_layout()
plt.savefig('cfo_y_histogram_unnormalized.png', dpi=300, bbox_inches='tight')
plt.show()

# === Unnormalized Global Violin Plot ===
plt.figure(figsize=(10, 6))
sns.violinplot(data=cfo_y_values_unnormalized, inner="box", linewidth=1.25)
plt.title('Violin Plot of CFO_y Values (Unnormalized)', fontsize=16)
plt.ylabel('CFO_y (Hz)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cfo_y_violinplot_unnormalized.png', dpi=300, bbox_inches='tight')
plt.show()

# === Unnormalized Class-wise Violin Plot ===
plt.figure(figsize=(14, 6))
sns.violinplot(data=[classwise_cfo_unnormalized[i] for i in range(num_classes)], inner="box", linewidth=1.25)
plt.title('Class-wise Violin Plot of CFO_y Values (Unnormalized)', fontsize=16)
plt.xlabel('Radio ID (Class)', fontsize=14)
plt.ylabel('CFO_y (Hz)', fontsize=14)
plt.xticks(ticks=list(range(num_classes)), labels=[f'Radio{i}' for i in range(num_classes)], rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cfo_y_classwise_violinplot_unnormalized.png', dpi=300)
plt.show()

# =============================================
# PRINT COMPREHENSIVE STATISTICS
# =============================================
print(f"\n" + "="*60)
print(f"COMPREHENSIVE CFO_Y STATISTICS")
print(f"="*60)

print(f"\n--- NORMALIZED CFO_y Statistics ---")
print(f"Mean CFO_y: {mean_cfo_norm:.6f}")
print(f"Std CFO_y: {std_cfo_norm:.6f}")
print(f"Min CFO_y: {np.min(cfo_y_values_normalized):.6f}")
print(f"Max CFO_y: {np.max(cfo_y_values_normalized):.6f}")
print(f"Median CFO_y: {np.median(cfo_y_values_normalized):.6f}")

print(f"\n--- UNNORMALIZED CFO_y Statistics ---")
print(f"Mean CFO_y: {mean_cfo_unnorm:.2f} Hz")
print(f"Std CFO_y: {std_cfo_unnorm:.2f} Hz") 
print(f"Min CFO_y: {np.min(cfo_y_values_unnormalized):.2f} Hz")
print(f"Max CFO_y: {np.max(cfo_y_values_unnormalized):.2f} Hz")
print(f"Median CFO_y: {np.median(cfo_y_values_unnormalized):.2f} Hz")

print(f"\n--- Normalization Factor ---")
print(f"max_cfo (normalization factor): {max_cfo} Hz")

print(f"\n--- Class-wise Statistics (Unnormalized) ---")
for class_id in range(num_classes):
    class_cfo_values = classwise_cfo_unnormalized[class_id]
    if len(class_cfo_values) > 0:
        mean_class = np.mean(class_cfo_values)
        std_class = np.std(class_cfo_values)
        min_class = np.min(class_cfo_values)
        max_class = np.max(class_cfo_values)
        print(f"Radio{class_id}: Mean={mean_class:.2f} Hz, Std={std_class:.2f} Hz, Min={min_class:.2f} Hz, Max={max_class:.2f} Hz, Samples={len(class_cfo_values)}")

print(f"\nPlots saved:")
print(f"- Normalized: cfo_y_histogram_normalized.png, cfo_y_violinplot_normalized.png, cfo_y_classwise_violinplot_normalized.png")
print(f"- Unnormalized: cfo_y_histogram_unnormalized.png, cfo_y_violinplot_unnormalized.png, cfo_y_classwise_violinplot_unnormalized.png")
