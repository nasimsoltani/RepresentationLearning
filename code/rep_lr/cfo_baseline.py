import os
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from dotenv import load_dotenv

# Adjust the import path to be relative to the project root
from py_datasets import TrainDataset

import os 

os.environ['PKL_DATASET_PATH'] = '/scratch/10608/aadharsh_aadhithya/data/rep_lr/OracleDatasetProcessed-arranged/rf_partition_dict_0.5.pkl'
#os.environ['PKL_DATASET_PATH'] = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl'

class SimpleNN(nn.Module):
    """
    A deeper neural network for regression with batch normalization and dropout.
    """
    def __init__(self, input_dim, hidden_dims=[ 512,128,64, 32], dropout=0.2):
        super(SimpleNN, self).__init__()
        
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
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Kaiming initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

def get_data(args):
    """
    Loads and processes data from the specified pickle file.
    """
    with open(args.pkl_dataset_path, 'rb') as handle:
        content = pickle.load(handle)
    
    train_list = content['train']
    val_list = content['val']
    test_list = content['test']
    mean_cfo = content['mean_cfo']
    std_cfo = content['std_cfo']
    
    # The dataset class requires an `args` object with `slice_len`, 
    # but it's not used for CFO data.
    dummy_args = argparse.Namespace(slice_len=0) 

    # We set test_mode=False for all to avoid data augmentation (slicing) on RF signals,
    # ensuring we get one sample per file.
    train_dataset = TrainDataset(train_list, None, dummy_args, None, mean_cfo, std_cfo, test_mode=False)
    val_dataset = TrainDataset(val_list, None, dummy_args, None, mean_cfo, std_cfo, test_mode=False)
    test_dataset = TrainDataset(test_list, None, dummy_args, None, mean_cfo, std_cfo, test_mode=False)

    def extract_cfo_data(dataset, name):
        X, y, filenames = [], [], []
        for i in tqdm(range(len(dataset)), desc=f"Extracting {name} data"):
            try:
                _, _, cfo_x, cfo_y, _, _, fpath = dataset[i]
                X.append(cfo_x.flatten().numpy())
                y.append(cfo_y.item())
                filenames.append(os.path.basename(fpath))
            except Exception as e:
                print(f"Skipping a sample due to error: {e}")
                continue
        return np.array(X), np.array(y), filenames

    print("Loading training data...")
    X_train, y_train, _ = extract_cfo_data(train_dataset, "train")
    print("Loading validation data...")
    X_val, y_val, _ = extract_cfo_data(val_dataset, "validation")
    print("Loading test data...")
    X_test, y_test, test_filenames = extract_cfo_data(test_dataset, "test")

    # Create DataLoaders for the PyTorch NN model
    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1)), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float().unsqueeze(1)), batch_size=args.batch_size)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test, test_filenames), (train_dl, val_dl), (mean_cfo, std_cfo)

def train_nn(model, train_dl, val_dl, args, device):
    """
    Training loop for the neural network with early stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dl)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dl)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_nn_model.pt'))
            print("Saved best model.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {args.patience} epochs.")
                break

def evaluate(model, X_test, y_test, test_filenames, mean_cfo, std_cfo, args, device=None):
    """
    Evaluates the model and saves metrics and predictions.
    """
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test).float().to(device)
            y_pred_normalized = model(inputs).cpu().numpy().flatten()
    else:  # scikit-learn model
        y_pred_normalized = model.predict(X_test)
        
    y_true_normalized = y_test

    # De-normalize predictions and ground truth to their real values
    y_true_real = (y_true_normalized * std_cfo) + mean_cfo
    y_pred_real = (y_pred_normalized * std_cfo) + mean_cfo

    # Calculate metrics
    metrics = {
        'mae': float(mean_absolute_error(y_true_real, y_pred_real)),
        'mse': float(mean_squared_error(y_true_real, y_pred_real)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_real, y_pred_real))),
        'r2_score': float(r2_score(y_true_real, y_pred_real))
    }
    
    print("\nCFO Estimation Metrics:")
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")
        
    # Save metrics to a file
    metrics_path = os.path.join(args.output_dir, f'{args.model}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    # Save detailed predictions to a file
    predictions = []
    for i in range(len(test_filenames)):
        predictions.append({
            'filename': test_filenames[i],
            'y_true_normalized': float(y_true_normalized[i]),
            'y_pred_normalized': float(y_pred_normalized[i]),
            'y_true_real': float(y_true_real[i]),
            'y_pred_real': float(y_pred_real[i]),
        })
        
    predictions_path = os.path.join(args.output_dir, f'{args.model}_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {predictions_path}")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='CFO Estimation Baseline Experiments')
    parser.add_argument('--pkl_dataset_path', type=str, default=os.getenv('PKL_DATASET_PATH'), help='Path to the pkl dataset file. Reads from PKL_DATASET_PATH environment variable if not provided.')
    parser.add_argument('--model', type=str, choices=['nn', 'linear', 'svr', 'kernel_ridge'], required=True, help='Model to train and evaluate.')
    parser.add_argument('--output_dir', type=str, default='results_cfo_baseline', help='Directory to save results.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for NN.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for NN training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for NN training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for NN training.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping in NN training.')
    args = parser.parse_args()

    if not args.pkl_dataset_path:
        raise ValueError("PKL_DATASET_PATH environment variable not set, and --pkl_dataset_path argument not provided.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    (X_train, y_train), (X_val, y_val), (X_test, y_test, test_filenames), (train_dl, val_dl), (mean_cfo, std_cfo) = get_data(args)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    if args.model == 'nn':
        input_dim = X_train.shape[1]
        model = SimpleNN(input_dim).to(device)
        print("Training Neural Network...")
        train_nn(model, train_dl, val_dl, args, device)
        # Load best model for evaluation
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_nn_model.pt')))
        print("Evaluating Neural Network...")
        evaluate(model, X_test, y_test, test_filenames, mean_cfo, std_cfo, args, device)

    elif args.model == 'linear':
        print("Training Linear Model with GridSearchCV...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SGDRegressor(random_state=42, max_iter=2000, tol=1e-4, early_stopping=True))
        ])

        param_grid = {
            'regressor__loss': ['squared_error', 'huber'],
            'regressor__penalty': ['l2', 'l1'],
            'regressor__alpha': np.logspace(-4, 0, 5),
            'regressor__epsilon': [0.1, 0.2]  # Only used by huber loss
        }

        search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        search.fit(X_train, y_train)

        print("\nBest parameters found for Linear Model:")
        print(search.best_params_)
        
        model = search.best_estimator_
        print("\nEvaluating Best Linear Model...")
        evaluate(model, X_test, y_test, test_filenames, mean_cfo, std_cfo, args)
    
    elif args.model == 'svr':
        print("Training SVR with RandomizedSearchCV...")
        
        # SVR can be very slow, so we'll use a subset of the training data.
        subset_size = 5000
        if len(X_train) > subset_size:
            print(f"SVR training is slow. Using a random subset of {subset_size} samples for hyperparameter search.")
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            X_train_sub, y_train_sub = X_train[indices], y_train[indices]
        else:
            X_train_sub, y_train_sub = X_train, y_train

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        
        param_dist = {
            'svr__C': np.logspace(-2, 2, 5),
            'svr__gamma': np.logspace(-3, 1, 5),
            'svr__kernel': ['rbf']
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
        search.fit(X_train_sub, y_train_sub)

        print("\nBest parameters found for SVR:")
        print(search.best_params_)

        model = search.best_estimator_
        
        print("\nEvaluating SVR...")
        evaluate(model, X_test, y_test, test_filenames, mean_cfo, std_cfo, args)

    elif args.model == 'kernel_ridge':
        print("Training Kernel Ridge Regression with RandomizedSearchCV...")

        # Kernel Ridge can also be slow, so use a subset for tuning
        subset_size = 5000
        if len(X_train) > subset_size:
            print(f"Kernel Ridge training is slow. Using a random subset of {subset_size} samples for hyperparameter search.")
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            X_train_sub, y_train_sub = X_train[indices], y_train[indices]
        else:
            X_train_sub, y_train_sub = X_train, y_train

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('krr', KernelRidge())
        ])

        param_dist = {
            'krr__alpha': np.logspace(-2, 2, 5),
            'krr__kernel': ['rbf', 'linear', 'polynomial'],
            'krr__gamma': np.logspace(-3, 1, 5),
            'krr__degree': [2, 3]
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
        search.fit(X_train_sub, y_train_sub)

        print("\nBest parameters found for Kernel Ridge Regression:")
        print(search.best_params_)

        model = search.best_estimator_
        
        print("\nEvaluating Kernel Ridge Regression...")
        evaluate(model, X_test, y_test, test_filenames, mean_cfo, std_cfo, args)

if __name__ == '__main__':
    main() 