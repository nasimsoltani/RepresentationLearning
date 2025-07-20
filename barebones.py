import os
import json
import argparse
import pickle
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the necessary components from the models
from code.rep_lr.models import TaskAdaptiveEncoder, ComplexSequenceProjector, CFOEstimationHead
from code.rep_lr.py_datasets import TrainDataset


def train_epoch(model, train_dl, loss_fn, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(train_dl, desc="Training")):
        # Extract CFO data (indices 2, 3 are CFO_X, CFO_y)
        inputs = batch[2].to(device).float()  # CFO_X: (batch_size, 2, 160)
        labels = batch[3].to(device).float().view(-1, 1)  # CFO_y: (batch_size, 1)
        
        optimizer.zero_grad()
        
        # Forward pass: projection -> encoder -> head
        x = model['projection'](inputs)  # (batch_size, 2, 160) -> (batch_size, 2, proj_seq_len)
        x = model['encoder'](x)          # (batch_size, 2, proj_seq_len) -> (batch_size, 2, d2)
        outputs = model['head'](x)       # (batch_size, 2, d2) -> (batch_size, 1)
        
        loss = loss_fn(outputs, labels)
        
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected in batch {batch_idx}, skipping...")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def validate_epoch(model, val_dl, loss_fn, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Validating"):
            # Extract CFO data
            inputs = batch[2].to(device).float()
            labels = batch[3].to(device).float().view(-1, 1)
            
            # Forward pass
            x = model['projection'](inputs)
            x = model['encoder'](x)
            outputs = model['head'](x)
            
            loss = loss_fn(outputs, labels)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / max(num_batches, 1)


def evaluate_cfo_estimation(model, test_dl, device, output_dir, mean_cfo, std_cfo):
    """Evaluate CFO estimation and save results"""
    model.eval()
    y_true = []
    y_pred = []
    evaluation_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating CFO Estimation"):
            # Extract CFO data and file paths
            inputs = batch[2].to(device).float()  # CFO_X
            labels = batch[3].to(device).float()  # CFO_y
            file_paths = batch[6]  # file paths
            
            # Forward pass
            x = model['projection'](inputs)
            x = model['encoder'](x)
            outputs = model['head'](x)
            
            true_val = labels.item()
            pred_val = outputs.item()
            
            y_true.append(true_val)
            y_pred.append(pred_val)
            
            # Store detailed result
            file_path = file_paths[0]
            filename = os.path.basename(file_path)
            evaluation_results.append({
                'filename': filename,
                'y_true_normalized': true_val,
                'y_pred_normalized': pred_val,
                'y_true_real': true_val * std_cfo + mean_cfo,
                'y_pred_real': pred_val * std_cfo + mean_cfo,
            })
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # De-normalize predictions and ground truth
    y_true_real = y_true * std_cfo + mean_cfo
    y_pred_real = y_pred * std_cfo + mean_cfo
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_real, y_pred_real)
    mse = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_real, y_pred_real)
    
    # Calculate normalized metrics
    mae_normalized = mean_absolute_error(y_true, y_pred)
    mse_normalized = mean_squared_error(y_true, y_pred)
    r2_normalized = r2_score(y_true, y_pred)
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'mae_normalized': float(mae_normalized),
        'mse_normalized': float(mse_normalized),
        'r2_normalized': float(r2_normalized)
    }
    
    print("\nCFO Estimation Metrics:")
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")
    
    # Save metrics to a JSON file
    metrics_path = os.path.join(output_dir, 'cfo_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save detailed predictions
    predictions_path = os.path.join(output_dir, 'cfo_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Detailed predictions saved to {predictions_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Barebones CFO Estimation with TaskAdaptiveEncoder (1 block)')
    parser.add_argument('--pkl_dataset_path', type=str, required=True, help='Path to the pkl dataset file.')
    parser.add_argument('--output_dir', type=str, default='results_barebones', help='Output directory for results.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--proj_seq_len', type=int, default=256, help='Projected sequence length.')
    parser.add_argument('--proj_hidden_dim', type=int, default=512, help='Projection hidden dimension.')
    parser.add_argument('--d2', type=int, default=128, help='Encoder output dimension.')
    parser.add_argument('--head_hidden_dim', type=int, default=256, help='Head hidden dimension.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'barebones_cfo_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")
    
    # Save args
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load data
    with open(args.pkl_dataset_path, 'rb') as handle:
        content = pickle.load(handle)
    
    train_list = content['train']
    val_list = content['val']
    test_list = content['test']
    max_cfo = content['max_cfo']
    mean_cfo = content['mean_cfo']
    std_cfo = content['std_cfo']
    
    print(f"Dataset loaded: {len(train_list)} train, {len(val_list)} val, {len(test_list)} test samples")
    
    # Create datasets
    dataset_args = argparse.Namespace(slice_len=1024)  # Not used for CFO but required by TrainDataset
    train_dataset = TrainDataset(train_list, {}, dataset_args, max_cfo, mean_cfo, std_cfo)
    val_dataset = TrainDataset(val_list, {}, dataset_args, max_cfo, mean_cfo, std_cfo)
    test_dataset = TrainDataset(test_list, {}, dataset_args, max_cfo, mean_cfo, std_cfo, test_mode=True)
    
    # Create data loaders
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    projection = ComplexSequenceProjector(
        input_seq_len=160,  # CFO input length
        output_seq_len=args.proj_seq_len,
        hidden_dim=args.proj_hidden_dim
    )
    
    encoder = TaskAdaptiveEncoder(
        slice_size=args.proj_seq_len,
        output_dim=args.d2,
        dropout=args.dropout,
        num_blocks=1  # Key requirement: just 1 block
    )
    
    head = CFOEstimationHead(
        input_dim=2 * args.d2,
        hidden_dim=args.head_hidden_dim,
        dropout=args.dropout
    )
    
    model = nn.ModuleDict({
        'projection': projection,
        'encoder': encoder,
        'head': head
    })
    
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture:")
    print(f"- Projection parameters: {sum(p.numel() for p in projection.parameters()):,}")
    print(f"- Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"- Head parameters: {sum(p.numel() for p in head.parameters()):,}")
    print(f"- Total parameters: {total_params:,}")
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.HuberLoss(delta=0.1)  # Robust to outliers
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_dl, loss_fn, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_dl, loss_fn, device)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(run_dir, 'best_model.pt'))
            
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model for evaluation
    print(f"\nLoading best model (val_loss: {best_val_loss:.4f})")
    checkpoint = torch.load(os.path.join(run_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_cfo_estimation(model, test_dl, device, run_dir, mean_cfo, std_cfo)
    
    # Save training summary
    summary = {
        'training_completed': True,
        'epochs_trained': checkpoint['epoch'],
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(checkpoint['train_loss']),
        'test_metrics': test_metrics,
        'model_config': {
            'projection_type': 'ComplexSequenceProjector',
            'encoder_type': 'TaskAdaptiveEncoder',
            'encoder_blocks': 1,
            'head_type': 'CFOEstimationHead',
            'total_parameters': total_params
        }
    }
    
    with open(os.path.join(run_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nTraining completed! Results saved to {run_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test R² score: {test_metrics['r2_score']:.4f}")


if __name__ == '__main__':
    main() 