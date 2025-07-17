import os
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

from py_datasets import TrainDataset

os.environ['PKL_DATASET_PATH'] = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl'

class ClassificationNN(nn.Module):
    """
    A neural network for binary classification.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(ClassificationNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

def get_data(args):
    """
    Loads data and creates binary labels for outlier detection.
    """
    with open(args.pkl_dataset_path, 'rb') as handle:
        content = pickle.load(handle)
    
    train_list, val_list, test_list = content['train'], content['val'], content['test']
    
    dummy_args = argparse.Namespace(slice_len=0)

    def extract_data(dataset_list, name, outlier_threshold):
        X, y_real, y_binary = [], [], []
        base_dataset = TrainDataset(dataset_list, None, dummy_args, None, 0, 1, test_mode=True)
        for i in tqdm(range(len(base_dataset)), desc=f"Extracting {name} data"):
            _, _, cfo_x, cfo_y, _, _, _ = base_dataset[i]
            X.append(cfo_x.flatten().numpy())
            y_real.append(cfo_y.item())
            y_binary.append(1 if cfo_y.item() < outlier_threshold else 0)
        return np.array(X), np.array(y_real), np.array(y_binary)

    print("Loading data...")
    X_train, y_train_real, y_train_binary = extract_data(train_list, "train", args.outlier_threshold)
    X_val, _, y_val_binary = extract_data(val_list, "validation", args.outlier_threshold)
    X_test, y_test_real, y_test_binary = extract_data(test_list, "test", args.outlier_threshold)

    # Calculate class weights for handling imbalance
    num_neg = np.sum(y_train_binary == 0)
    num_pos = np.sum(y_train_binary == 1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float) if num_pos > 0 else torch.tensor([1.0], dtype=torch.float)

    print(f"\nTraining data: {num_neg} normal, {num_pos} outliers.")
    print(f"Calculated pos_weight for loss function: {pos_weight.item():.2f}")

    # Create DataLoaders
    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_binary).float().unsqueeze(1)), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val_binary).float().unsqueeze(1)), batch_size=args.batch_size)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test_binary).float().unsqueeze(1)), batch_size=args.batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, y_test_real, pos_weight

def train_nn(model, train_dl, val_dl, pos_weight, args, device):
    """
    Training loop for the classification neural network.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item()
                all_preds.append(torch.sigmoid(outputs).cpu())
                all_labels.append(labels.cpu())
        
        avg_val_loss = val_loss / len(val_dl)
        auprc = average_precision_score(torch.cat(all_labels), torch.cat(all_preds))
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val AUPRC: {auprc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_outlier_model.pt'))
            print("Saved best model.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

def evaluate_and_plot(model, test_dl, y_test_real, args, device):
    """
    Evaluates the model and generates plots.
    """
    model.eval()
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_dl, desc="Evaluating on test set"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            y_true.extend(labels.numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # --- Metrics ---
    auprc = average_precision_score(y_true, y_pred_probs)
    auc_roc = roc_auc_score(y_true, y_pred_probs)
    print(f"\nTest Metrics:\nAUPRC: {auprc:.4f}\nAUC-ROC: {auc_roc:.4f}")

    # --- Plot 1: CFO Y-value distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test_real, bins=100, kde=True)
    plt.axvline(args.outlier_threshold, color='r', linestyle='--', label=f'Outlier Threshold ({args.outlier_threshold})')
    plt.title('Distribution of CFO Y-values in Test Set')
    plt.xlabel('CFO Value (Hz)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'cfo_y_distribution.png'))
    plt.close()

    # --- Plot 2: PR Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, label=f'AUPRC = {auprc:.4f}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'pr_curve.png'))
    plt.close()

    # --- Plot 3: Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Outlier'], yticklabels=['Normal', 'Outlier'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()

    # --- Plot 4: Distribution of CFO values for predicted outliers ---
    predicted_outlier_indices = np.where(y_pred_binary == 1)[0]
    predicted_outlier_y_values = y_test_real[predicted_outlier_indices]
    
    if len(predicted_outlier_y_values) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(predicted_outlier_y_values, bins=50, kde=True, color='orange')
        plt.axvline(args.outlier_threshold, color='r', linestyle='--', label=f'Outlier Threshold ({args.outlier_threshold})')
        plt.title('Distribution of CFO Y-values for Predicted Outliers')
        plt.xlabel('CFO Value (Hz)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'predicted_outliers_distribution.png'))
        plt.close()
    else:
        print("No outliers were predicted, skipping the predicted outlier distribution plot.")

    print(f"\nPlots saved to {args.output_dir}")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='CFO Outlier Detection')
    parser.add_argument('--pkl_dataset_path', type=str, default=os.getenv('PKL_DATASET_PATH'))
    parser.add_argument('--output_dir', type=str, default='results_cfo_outlier', help='Directory to save results.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--outlier_threshold', type=float, default=-5000.0, help='CFO Y-value threshold to define outliers.')
    args = parser.parse_args()

    if not args.pkl_dataset_path:
        raise ValueError("PKL_DATASET_PATH environment variable not set or provided.")
    
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    train_dl, val_dl, test_dl, y_test_real, pos_weight = get_data(args)
    
    input_dim = next(iter(train_dl))[0].shape[1]
    model = ClassificationNN(input_dim).to(device)
    
    print("\nStarting training...")
    train_nn(model, train_dl, val_dl, pos_weight, args, device)
    
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_outlier_model.pt')))
    
    evaluate_and_plot(model, test_dl, y_test_real, args, device)

if __name__ == '__main__':
    main() 