import os
import json
import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import random

from models import (LightProjectionLayer, ProjectionLayerMLP, Encoder, 
                   RFClassificationHead, ChannelEstimationHead, CFOEstimationHead)
from py_datasets import TrainDataset
from torch.utils.data import DataLoader

def evaluate_rf_fingerprinting(model, test_dl, device, output_dir, class_names):
    projection, encoder, task_head = model
    y_true = []
    y_pred = []
    evaluation_results = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            inputs, labels, _, _, _, _, file_paths = batch
            inputs = inputs.squeeze(0).to(device) # Squeeze batch dim, move slices to device
            labels = labels.to(device)

            # Get predictions for all slices
            slice_outputs = []
            # Process slices in mini-batches to avoid OOM
            mini_batch_size = 128 
            for i in range(0, inputs.size(0), mini_batch_size):
                input_slices = inputs[i:i+mini_batch_size]
                x = projection(input_slices)
                x = encoder(x)
                output = task_head(x)
                slice_outputs.append(output)
            
            all_slice_outputs = torch.cat(slice_outputs)
            
            # Aggregate predictions (majority vote)
            _, predicted_slices = torch.max(all_slice_outputs, 1)
            final_prediction = torch.mode(predicted_slices).values.item()

            y_true.append(labels.item())
            y_pred.append(final_prediction)

            # Store detailed result
            file_path = file_paths[0]
            filename = os.path.basename(file_path)
            evaluation_results.append({
                'filename': filename,
                'pred_class': final_prediction,
                'gt_class': labels.item()
            })

    # Save detailed predictions to a JSON file
    predictions_path = os.path.join(output_dir, 'predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"\nDetailed predictions saved to {predictions_path}")

    # Calculate and print metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save report to file
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")
    plt.close()

def evaluate_cfo_estimation(model, test_dl, device, output_dir, max_cfo):
    projection, encoder, task_head = model
    y_true = []
    y_pred = []
    evaluation_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating CFO Estimation"):
            # Data for CFO is at index 2 (CFO_X) and 3 (CFO_y)
            _, _, inputs, labels, _, _, file_paths = batch
            
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            
            x = projection(inputs)
            x = encoder(x)
            outputs = task_head(x)
            
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
                'y_true_real': true_val * max_cfo,
                'y_pred_real': pred_val * max_cfo,
            })
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # De-normalize predictions and ground truth
    y_true_real = y_true * max_cfo
    y_pred_real = y_pred * max_cfo
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_real, y_pred_real)
    mse = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_real, y_pred_real)
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2)
    }
    
    print("\nCFO Estimation Metrics:")
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")
        
    # Save metrics to a JSON file
    metrics_path = os.path.join(output_dir, 'cfo_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save detailed predictions with filenames
    predictions_path = os.path.join(output_dir, 'cfo_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Detailed predictions saved to {predictions_path}")
    
    #TODO: See what is a meaningfull plot
    # Plotting
    # plt.figure(figsize=(8, 8))
    # plt.scatter(y_true_real, y_pred_real, alpha=0.5)
    # plt.plot([min(y_true_real), max(y_true_real)], [min(y_true_real), max(y_true_real)], 'r--')
    # plt.xlabel('True CFO (Hz)')
    # plt.ylabel('Predicted CFO (Hz)')
    # plt.title('CFO Estimation: Predicted vs. True')
    # plt.grid(True)
    # plot_path = os.path.join(output_dir, 'cfo_pred_vs_true.png')
    # plt.savefig(plot_path)
    # print(f"Scatter plot saved to {plot_path}")
    # plt.close()

def evaluate_channel_estimation(model, test_dl, device, output_dir):
    projection, encoder, task_head = model
    all_y_true = []
    all_y_pred = []
    evaluation_results = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating Channel Estimation"):
            # Data for Channel is at index 4 (Channel_X) and 5 (Channel_y)
            _, _, _, _, inputs, labels, file_paths = batch

            inputs = inputs.to(device).float()
            labels = labels.to(device) # Shape: (B, 2, 52)

            x = projection(inputs)
            x = encoder(x)
            outputs = task_head(x) # Shape: (B, 2, 52)

            true_val = labels.cpu().numpy()[0]
            pred_val = outputs.cpu().numpy()[0]
            
            all_y_true.append(true_val)
            all_y_pred.append(pred_val)

            # Store detailed result
            file_path = file_paths[0]
            filename = os.path.basename(file_path)
            evaluation_results.append({
                'filename': filename,
                'y_true': true_val.tolist(),
                'y_pred': pred_val.tolist(),
            })

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    # Separate real and imaginary parts
    y_true_real = y_true[:, 0, :]
    y_pred_real = y_pred[:, 0, :]
    y_true_imag = y_true[:, 1, :]
    y_pred_imag = y_pred[:, 1, :]

    # Convert to complex numbers for NMSE
    y_true_complex = y_true_real + 1j * y_true_imag
    y_pred_complex = y_pred_real + 1j * y_pred_imag

    # Calculate NMSE
    nmse_num = np.mean(np.abs(y_true_complex - y_pred_complex)**2)
    nmse_den = np.mean(np.abs(y_true_complex)**2)
    nmse = nmse_num / nmse_den
    
    # Calculate NMSE in dB
    nmse_db = 10 * np.log10(nmse)
    
    # Calculate metrics for real and imaginary parts separately
    mae_real = mean_absolute_error(y_true_real, y_pred_real)
    mse_real = mean_squared_error(y_true_real, y_pred_real)
    r2_real = r2_score(y_true_real, y_pred_real)

    mae_imag = mean_absolute_error(y_true_imag, y_pred_imag)
    mse_imag = mean_squared_error(y_true_imag, y_pred_imag)
    r2_imag = r2_score(y_true_imag, y_pred_imag)

    metrics = {
        'nmse': float(nmse),
        'nmse_db': float(nmse_db),
        'real_part': {
            'mae': float(mae_real),
            'mse': float(mse_real),
            'r2_score': float(r2_real)
        },
        'imaginary_part': {
            'mae': float(mae_imag),
            'mse': float(mse_imag),
            'r2_score': float(r2_imag)
        }
    }
    
    print("\nChannel Estimation Metrics:")
    print(f"NMSE: {nmse:.4f}")
    print(f"NMSE (dB): {nmse_db:.4f}")
    print("\nReal Part Metrics:")
    for key, value in metrics['real_part'].items():
        print(f"{key.upper()}: {value:.4f}")
    print("\nImaginary Part Metrics:")
    for key, value in metrics['imaginary_part'].items():
        print(f"{key.upper()}: {value:.4f}")

    # Save metrics to a JSON file
    metrics_path = os.path.join(output_dir, 'channel_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save detailed predictions
    predictions_path = os.path.join(output_dir, 'channel_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Detailed predictions saved to {predictions_path}")
    
    #TODO: Decide what is a meaningfull plot
    # Save some examples for plotting
    # num_examples_to_save = min(5, len(y_true))
    # example_indices = np.random.choice(len(y_true), num_examples_to_save, replace=False)
    
    # plot_data = {
    #     'true': y_true[example_indices].tolist(),
    #     'pred': y_pred[example_indices].tolist()
    # }
    # plot_data_path = os.path.join(output_dir, 'channel_plot_data.json')
    # with open(plot_data_path, 'w') as f:
    #     json.dump(plot_data, f, indent=4)

    # # Plotting examples
    # if num_examples_to_save > 0:
    #     fig, axes = plt.subplots(num_examples_to_save, 2, figsize=(12, 4 * num_examples_to_save), squeeze=False)
    #     fig.suptitle('Channel Estimation Examples (Magnitude and Phase)', fontsize=16)

    #     for i, idx in enumerate(example_indices):
    #         true_mag = np.abs(y_true_complex[idx])
    #         pred_mag = np.abs(y_pred_complex[idx])
    #         true_phase = np.angle(y_true_complex[idx])
    #         pred_phase = np.angle(y_pred_complex[idx])

    #         # Magnitude plot
    #         axes[i, 0].plot(true_mag, 'b-', label='True Magnitude')
    #         axes[i, 0].plot(pred_mag, 'r--', label='Predicted Magnitude')
    #         axes[i, 0].set_title(f'Example {i+1} - Magnitude')
    #         axes[i, 0].set_xlabel('Subcarrier Index')
    #         axes[i, 0].set_ylabel('Magnitude')
    #         axes[i, 0].legend()
    #         axes[i, 0].grid(True)

    #         # Phase plot
    #         axes[i, 1].plot(true_phase, 'b-', label='True Phase')
    #         axes[i, 1].plot(pred_phase, 'r--', label='Predicted Phase')
    #         axes[i, 1].set_title(f'Example {i+1} - Phase')
    #         axes[i, 1].set_xlabel('Subcarrier Index')
    #         axes[i, 1].set_ylabel('Phase (radians)')
    #         axes[i, 1].legend()
    #         axes[i, 1].grid(True)
            
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     plot_path = os.path.join(output_dir, 'channel_estimation_examples.png')
    #     plt.savefig(plot_path)
    #     print(f"Example plots saved to {plot_path}")
    #     plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluation script for representation learning models.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--task', type=str, required=True, choices=['rf_fingerprinting', 'channel_estimation', 'cfo_estimation'], help='Task to evaluate.')
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used.')
    parser.add_argument('--test_fraction', type=float, default=1.0, help='Fraction of the test set to use for evaluation.')
    cli_args = parser.parse_args()

    # Determine model directory
    model_dir = os.path.dirname(cli_args.model_path)

    # Load args from the saved json file
    args_path = os.path.join(model_dir, 'args.json')
    with open(args_path, 'r') as f:
        args = argparse.Namespace(**json.load(f))

    # Set device
    device = torch.device(f'cuda:{cli_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    with open(args.pkl_dataset_path, 'rb') as handle:
        content = pickle.load(handle)
    test_list = content['test']
    max_cfo = content['max_cfo']
    
    # Shuffle and subset the test set
    random.shuffle(test_list)
    num_test_samples = int(len(test_list) * cli_args.test_fraction)
    test_list = test_list[:num_test_samples]
    print(f"Using {num_test_samples} samples from the test set ({cli_args.test_fraction*100:.2f}%).")
    
    # Create ID class dict 
    ID_class_dict = {}
    for i in range(16):
        this_key = 'Radio'+str(i)
        ID_class_dict[this_key] = i
    num_classes = len(list(ID_class_dict.keys()))

    test_dataset = TrainDataset(test_list, ID_class_dict, args, max_cfo, test_mode=True)
    # Use batch_size=1 for test loader because of variable number of slices
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Re-create model architecture
    seq_len = args.slice_len if args.task == 'rf_fingerprinting' else 160
    if args.projection_type == 'light':
        projection = LightProjectionLayer(in_channels=2, out_channels=args.proj_channels, output_dim=args.d1)
    else:
        projection = ProjectionLayerMLP(in_channels=2, out_channels=args.proj_channels, seq_length=seq_len, output_dim=args.d1)
    
    encoder = Encoder(input_dim=args.d1, hidden_dims=args.encoder_hidden_dims, output_dim=args.d2, dropout=args.dropout)

    if cli_args.task == 'rf_fingerprinting':
        task_head = RFClassificationHead(input_dim=args.d2, num_classes=num_classes, hidden_dim=args.head_hidden_dim, dropout=args.dropout)
    elif cli_args.task == 'channel_estimation':
        task_head = ChannelEstimationHead(input_dim=args.d2, hidden_dim=args.head_hidden_dim, output_length=52, dropout=args.dropout)
    elif cli_args.task == 'cfo_estimation':
        task_head = CFOEstimationHead(input_dim=args.d2, hidden_dim=args.head_hidden_dim, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown task: {cli_args.task}")

    model = torch.nn.ModuleList([projection, encoder, task_head])
    for module in model:
        module.to(device)

    # Load trained weights
    checkpoint = torch.load(cli_args.model_path, map_location=device)
    for i, module in enumerate(model):
        module.load_state_dict(checkpoint[f'module_{i}'])
    
    for module in model:
        module.eval()

    print("Model loaded successfully.")

    # Task head validation
    task_head = model[2]
    task_valid = False
    if cli_args.task == 'rf_fingerprinting' and isinstance(task_head, RFClassificationHead):
        task_valid = True
    elif cli_args.task == 'cfo_estimation' and isinstance(task_head, CFOEstimationHead):
        task_valid = True
    elif cli_args.task == 'channel_estimation' and isinstance(task_head, ChannelEstimationHead):
        task_valid = True

    if not task_valid:
        raise ValueError(f"Mismatched task and model head. Task: {cli_args.task}, Head: {type(task_head).__name__}")

    # Evaluation loop
    if cli_args.task == 'rf_fingerprinting':
        evaluate_rf_fingerprinting(model, test_dl, device, model_dir, list(ID_class_dict.keys()))
    elif cli_args.task == 'cfo_estimation':
        evaluate_cfo_estimation(model, test_dl, device, model_dir, max_cfo)
    elif cli_args.task == 'channel_estimation':
        evaluate_channel_estimation(model, test_dl, device, model_dir)

if __name__ == '__main__':
    main() 