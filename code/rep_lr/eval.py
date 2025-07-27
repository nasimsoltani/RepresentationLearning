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
import re

from models import (ComplexSequenceProjector, UpsamplingProjector, Encoder, 
                   RFClassificationHead, ChannelEstimationHead, CFOEstimationHead,
                   SimpleCFOEstimationHead, DirectCFOEstimationHead, CFOAdaptiveHead, TaskAdaptiveEncoder)
from py_datasets import TrainDataset, TrainDatasetRFixed
from torch.utils.data import DataLoader

def evaluate_rf_fingerprinting(model, test_dl, device, output_dir, class_names, args):
    is_mtl = getattr(args, 'mtl', False)
    
    y_true = []
    y_pred = []
    evaluation_results = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            inputs, labels, cfo_inputs_raw, _, channel_inputs_raw, _, file_paths = batch
            labels = labels.to(device)

            # Pre-calculate projections from other tasks for MTL
            projected_others = 0
            if is_mtl:
                task_data_map = {
                    'cfo_estimation': cfo_inputs_raw,
                    'channel_estimation': channel_inputs_raw
                }
                for task_name in args.task:
                    if task_name != 'rf_fingerprinting':
                        other_inputs = task_data_map[task_name].to(device).float()
                        projected_others += model['projections'][task_name](other_inputs)

            # Process slices in mini-batches to avoid OOM
            rf_inputs_all_slices = inputs.squeeze(0).to(device)
            slice_outputs = []
            mini_batch_size = 128
            for i in range(0, rf_inputs_all_slices.size(0), mini_batch_size):
                input_slices = rf_inputs_all_slices[i:i+mini_batch_size]

                if is_mtl:
                    projected_rf = model['projections']['rf_fingerprinting'](input_slices)
                    projected_sum = projected_rf + projected_others  # Broadcast
                    encoded = model['encoder'](projected_sum)
                    output = model['heads']['rf_fingerprinting'](encoded)
                else:
                    x = model['projection'](input_slices)
                    x = model['encoder'](x)
                    output = model['head'](x)
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

    # Plot distance vs accuracy
    plot_distance_vs_accuracy(predictions_path)

def plot_distance_vs_accuracy(predictions_path):
    """
    Plots distance vs accuracy based on predictions from an RF fingerprinting task.

    Args:
        predictions_path (str): Path to the predictions JSON file.
    """
    try:
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {predictions_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {predictions_path}")
        return

    # Group predictions by distance
    distance_data = {}
    for item in predictions:
        filename = item['filename']
        match = re.search(r'_(\d+)ft_', filename)
        if match:
            distance = int(match.group(1))
            if distance not in distance_data:
                distance_data[distance] = {'correct': 0, 'total': 0}
            
            if item['pred_class'] == item['gt_class']:
                distance_data[distance]['correct'] += 1
            distance_data[distance]['total'] += 1

    if not distance_data:
        print("No distances found in filenames. Skipping distance vs. accuracy plot.")
        return

    # Calculate accuracies and sort by distance
    sorted_distances = sorted(distance_data.keys())
    accuracies = [distance_data[d]['correct'] / distance_data[d]['total'] for d in sorted_distances]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, accuracies, marker='o', linestyle='-')
    plt.title('Distance vs. Accuracy')
    plt.xlabel('Distance (ft)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Save the plot
    output_dir = os.path.dirname(predictions_path)
    plot_path = os.path.join(output_dir, 'distance_vs_accuracy.png')
    plt.savefig(plot_path)
    print(f"Distance vs. accuracy plot saved to {plot_path}")
    plt.close()

def plot_distance_vs_cfo_metrics(predictions_path):
    """
    Plots distance vs CFO estimation metrics (MSE and R²).

    Args:
        predictions_path (str): Path to the CFO predictions JSON file.
    """
    try:
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"Error: CFO predictions file not found at {predictions_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {predictions_path}")
        return

    # Group predictions by distance
    distance_data = {}
    for item in predictions:
        filename = item['filename']
        match = re.search(r'_(\d+)ft_', filename)
        if match:
            distance = int(match.group(1))
            if distance not in distance_data:
                distance_data[distance] = {'y_true': [], 'y_pred': []}
            
            distance_data[distance]['y_true'].append(item['y_true_real'])
            distance_data[distance]['y_pred'].append(item['y_pred_real'])

    if not distance_data:
        print("No distances found in filenames. Skipping CFO distance vs. metrics plots.")
        return

    # Calculate metrics for each distance
    sorted_distances = sorted(distance_data.keys())
    mse_values = []
    r2_values = []
    
    for distance in sorted_distances:
        y_true = np.array(distance_data[distance]['y_true'])
        y_pred = np.array(distance_data[distance]['y_pred'])
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        mse_values.append(mse)
        r2_values.append(r2)

    output_dir = os.path.dirname(predictions_path)
    
    # Create MSE vs Distance plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, mse_values, marker='o', linestyle='-', color='red')
    plt.title('CFO Estimation: MSE vs. Distance')
    plt.xlabel('Distance (ft)')
    plt.ylabel('MSE (Hz²)')
    plt.grid(True)
    mse_plot_path = os.path.join(output_dir, 'cfo_distance_vs_mse.png')
    plt.savefig(mse_plot_path)
    print(f"CFO distance vs. MSE plot saved to {mse_plot_path}")
    plt.close()
    
    # Create R² vs Distance plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, r2_values, marker='o', linestyle='-', color='blue')
    plt.title('CFO Estimation: R² vs. Distance')
    plt.xlabel('Distance (ft)')
    plt.ylabel('R² Score')
    plt.grid(True)
    r2_plot_path = os.path.join(output_dir, 'cfo_distance_vs_r2.png')
    plt.savefig(r2_plot_path)
    print(f"CFO distance vs. R² plot saved to {r2_plot_path}")
    plt.close()

def plot_distance_vs_channel_metrics(predictions_path):
    """
    Plots distance vs channel estimation metrics (NMSE and R²).

    Args:
        predictions_path (str): Path to the channel predictions JSON file.
    """
    try:
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Channel predictions file not found at {predictions_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {predictions_path}")
        return

    # Group predictions by distance
    distance_data = {}
    for item in predictions:
        filename = item['filename']
        match = re.search(r'_(\d+)ft_', filename)
        if match:
            distance = int(match.group(1))
            if distance not in distance_data:
                distance_data[distance] = {'y_true': [], 'y_pred': []}
            
            distance_data[distance]['y_true'].append(item['y_true'])
            distance_data[distance]['y_pred'].append(item['y_pred'])

    if not distance_data:
        print("No distances found in filenames. Skipping channel distance vs. metrics plots.")
        return

    # Calculate metrics for each distance
    sorted_distances = sorted(distance_data.keys())
    nmse_values = []
    nmse_db_values = []
    r2_total_values = []
    
    for distance in sorted_distances:
        y_true = np.array(distance_data[distance]['y_true'])
        y_pred = np.array(distance_data[distance]['y_pred'])
        
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
        nmse_db = 10 * np.log10(nmse)
        
        # Calculate total R² by flattening both real and imaginary parts together
        y_true_flat = np.concatenate([y_true_real.flatten(), y_true_imag.flatten()])
        y_pred_flat = np.concatenate([y_pred_real.flatten(), y_pred_imag.flatten()])
        r2_total = r2_score(y_true_flat, y_pred_flat)
        
        nmse_values.append(nmse)
        nmse_db_values.append(nmse_db)
        r2_total_values.append(r2_total)

    output_dir = os.path.dirname(predictions_path)
    
    # Create NMSE vs Distance plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, nmse_values, marker='o', linestyle='-', color='red')
    plt.title('Channel Estimation: NMSE vs. Distance')
    plt.xlabel('Distance (ft)')
    plt.ylabel('NMSE')
    plt.grid(True)
    nmse_plot_path = os.path.join(output_dir, 'channel_distance_vs_nmse.png')
    plt.savefig(nmse_plot_path)
    print(f"Channel distance vs. NMSE plot saved to {nmse_plot_path}")
    plt.close()
    
    # Create NMSE (dB) vs Distance plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, nmse_db_values, marker='o', linestyle='-', color='orange')
    plt.title('Channel Estimation: NMSE (dB) vs. Distance')
    plt.xlabel('Distance (ft)')
    plt.ylabel('NMSE (dB)')
    plt.grid(True)
    nmse_db_plot_path = os.path.join(output_dir, 'channel_distance_vs_nmse_db.png')
    plt.savefig(nmse_db_plot_path)
    print(f"Channel distance vs. NMSE (dB) plot saved to {nmse_db_plot_path}")
    plt.close()
    
    # Create Total R² vs Distance plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, r2_total_values, marker='o', linestyle='-', color='blue')
    plt.title('Channel Estimation: Total R² vs. Distance')
    plt.xlabel('Distance (ft)')
    plt.ylabel('R² Score')
    plt.grid(True)
    r2_plot_path = os.path.join(output_dir, 'channel_distance_vs_r2.png')
    plt.savefig(r2_plot_path)
    print(f"Channel distance vs. total R² plot saved to {r2_plot_path}")
    plt.close()

def evaluate_cfo_estimation(model, test_dl, device, output_dir, max_cfo, mean_cfo, std_cfo, args):
    is_mtl = getattr(args, 'mtl', False)
    
    y_true = []
    y_pred = []
    evaluation_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating CFO Estimation"):
            rf_inputs_raw, _, inputs, labels, channel_inputs_raw, _, file_paths = batch
            
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            if is_mtl:
                # Average RF projections and sum with others
                projected_sum = 0
                task_data_map = {
                    'rf_fingerprinting': rf_inputs_raw,
                    'cfo_estimation': inputs,
                    'channel_estimation': channel_inputs_raw
                }
                for task_name in args.task:
                    task_inputs = task_data_map[task_name].to(device).float()
                    if task_name == 'rf_fingerprinting':
                        # Average projections across all slices for a single representation
                        proj = model['projections'][task_name](task_inputs.squeeze(0))
                        projected_sum += proj.mean(dim=0, keepdim=True)
                    else:
                        projected_sum += model['projections'][task_name](task_inputs)
                
                # For MTL, direct CFO is not typically used, but handle it just in case
                if getattr(args, 'direct_cfo', False):
                    # Extract just the CFO projection for direct processing
                    cfo_projection = model['projections']['cfo_estimation'](inputs)
                    outputs = model['heads']['cfo_estimation'](cfo_projection)
                else:
                    encoded = model['encoder'](projected_sum)
                    outputs = model['heads']['cfo_estimation'](encoded)
            else:
                x = model['projection'](inputs)
                
                # For single-task, check if using direct CFO (bypassing encoder)
                if getattr(args, 'direct_cfo', False):
                    outputs = model['head'](x)  # Direct: projection -> head
                else:
                    x = model['encoder'](x)          # Normal: projection -> encoder -> head
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
    # y_true_real = y_true * max_cfo
    # y_pred_real = y_pred * max_cfo
    y_true_real = y_true * std_cfo + mean_cfo
    y_pred_real = y_pred * std_cfo + mean_cfo
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_real, y_pred_real)
    mse = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_real, y_pred_real)

    #calculate normalized error
    mae_normalized = mean_absolute_error(y_true, y_pred)
    mse_normalized = mean_squared_error(y_true, y_pred)
    #rmse_normalized = np.sqrt(mse_normalized)
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
    
    # Save detailed predictions with filenames
    predictions_path = os.path.join(output_dir, 'cfo_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Detailed predictions saved to {predictions_path}")
    
    # Plot distance vs CFO metrics
    plot_distance_vs_cfo_metrics(predictions_path)
    
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

def evaluate_channel_estimation(model, test_dl, device, output_dir, args):
    is_mtl = getattr(args, 'mtl', False)

    
    all_y_true = []
    all_y_pred = []
    evaluation_results = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating Channel Estimation"):
            rf_inputs_raw, _, cfo_inputs_raw, _, inputs, labels, file_paths = batch

            inputs = inputs.to(device).float()
            labels = labels.to(device)

            if is_mtl:
                # Average RF projections and sum with others
                projected_sum = 0
                task_data_map = {
                    'rf_fingerprinting': rf_inputs_raw,
                    'cfo_estimation': cfo_inputs_raw,
                    'channel_estimation': inputs
                }
                for task_name in args.task:
                    task_inputs = task_data_map[task_name].to(device).float()
                    if task_name == 'rf_fingerprinting':
                        proj = model['projections'][task_name](task_inputs.squeeze(0))
                        projected_sum += proj.mean(dim=0, keepdim=True)
                    else:
                        projected_sum += model['projections'][task_name](task_inputs)

                encoded = model['encoder'](projected_sum)
                outputs = model['heads']['channel_estimation'](encoded)
            else:
                x = model['projection'](inputs)
                x = model['encoder'](x)
                outputs = model['head'](x)

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
    
    # Plot distance vs channel metrics
    plot_distance_vs_channel_metrics(predictions_path)
    
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
    parser.add_argument('--eval_pkl_dataset_path', type=str, default=None, help='Path to a specific pkl dataset file for evaluation. Overrides path in args.json.')
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used.')
    parser.add_argument('--test_fraction', type=float, default=1.0, help='Fraction of the test set to use for evaluation.')
    cli_args = parser.parse_args()

    # Determine model directory
    model_dir = os.path.dirname(cli_args.model_path)

    # Load args from the saved json file
    args_path = os.path.join(model_dir, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {model_dir}. Cannot determine model architecture.")
    with open(args_path, 'r') as f:
        train_args = argparse.Namespace(**json.load(f))
    
    # Override training args with evaluation-specific args from CLI
    train_args.gpu_id = cli_args.gpu_id
    train_args.test_fraction = cli_args.test_fraction

    # Set device
    device = torch.device(f'cuda:{train_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    eval_data_path = cli_args.eval_pkl_dataset_path
    if eval_data_path:
        print(f"Using provided evaluation dataset: {eval_data_path}")
    else:
        eval_data_path = train_args.pkl_dataset_path
        print(f"Using evaluation dataset from training args: {eval_data_path}")
    
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(f"Evaluation dataset not found at '{eval_data_path}'")

    with open(eval_data_path, 'rb') as handle:
        content = pickle.load(handle)
    test_list = content['test']
    max_cfo = content['max_cfo']
    mean_cfo = content['mean_cfo']
    std_cfo = content['std_cfo']
    
    # Shuffle and subset the test set
    random.shuffle(test_list)
    num_test_samples = int(len(test_list) * train_args.test_fraction)
    test_list = test_list[:num_test_samples]
    print(f"Using {num_test_samples} samples from the test set ({train_args.test_fraction*100:.2f}%).")
    
    # Create ID class dict 
    ID_class_dict = {}
    for i in range(16):
        this_key = 'Radio'+str(i)
        ID_class_dict[this_key] = i
    num_classes = len(list(ID_class_dict.keys()))

    if train_args.rf_fixed:
        test_dataset = TrainDatasetRFixed(test_list, ID_class_dict, train_args, max_cfo, mean_cfo, std_cfo, rf_begin_idx=train_args.rf_begin_idx, test_mode=True)
    else:
        test_dataset = TrainDataset(test_list, ID_class_dict, train_args, max_cfo, mean_cfo, std_cfo, test_mode=True)

    # Use batch_size=1 for test loader because of variable number of slices
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # After train_args is loaded and before model construction
    encoder_num_blocks = getattr(train_args, 'encoder_num_blocks', 1)

    # Re-create model architecture
    is_mtl = getattr(train_args, 'mtl', False)

    if is_mtl:
        print("Reconstructing MTL model architecture.")
        projections = torch.nn.ModuleDict()
        heads = torch.nn.ModuleDict()

        for task in train_args.task:
            if task == 'rf_fingerprinting':
                seq_len = train_args.slice_len
                projections[task] = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=train_args.proj_seq_len, hidden_dim=train_args.proj_hidden_dim)
                heads[task] = RFClassificationHead(input_dim=2*train_args.d2, num_classes=num_classes, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)
            
            elif task == 'channel_estimation':
                seq_len = 160
                projections[task] = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=train_args.proj_seq_len, hidden_dim=train_args.proj_hidden_dim)
                heads[task] = ChannelEstimationHead(input_dim=2*train_args.d2, hidden_dim=train_args.head_hidden_dim, output_length=52, dropout=train_args.dropout)

            elif task == 'cfo_estimation':
                seq_len = 160
                projections[task] = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=train_args.proj_seq_len, hidden_dim=train_args.proj_hidden_dim)
                
                # Choose CFO head based on training arguments
                if getattr(train_args, 'adaptive_cfo', False):
                    heads[task] = CFOAdaptiveHead(input_dim=2*train_args.d2, hidden_dim=128, dropout=0.1)
                elif getattr(train_args, 'simple_cfo', False):
                    heads[task] = SimpleCFOEstimationHead(input_dim=2*train_args.d2, hidden_dim=64, dropout=0.1)
                else:
                    heads[task] = CFOEstimationHead(input_dim=2*train_args.d2, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)

        # Choose encoder based on training arguments (MTL)
        if getattr(train_args, 'task_adaptive_encoder', False):
            print("Using task-adaptive encoder.")
            encoder = TaskAdaptiveEncoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout, num_blocks=encoder_num_blocks)
        else:
            encoder = Encoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout, num_blocks=encoder_num_blocks)
        
        model = torch.nn.ModuleDict({
            'projections': projections,
            'encoder': encoder,
            'heads': heads
        })
    else:
        print("Reconstructing single-task model architecture.")
        task_name = train_args.task
        
        # Create projection layer based on task
        if task_name == 'cfo_estimation':
            # Use ComplexSequenceProjector for CFO (updated from UpsamplingProjector)
            seq_len = 160  # CFO input length
            projection = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=train_args.proj_seq_len, hidden_dim=train_args.proj_hidden_dim)
        else:
            seq_len = train_args.slice_len if task_name == 'rf_fingerprinting' else 160
            projection = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=train_args.proj_seq_len, hidden_dim=train_args.proj_hidden_dim)
        
        # Choose encoder based on training arguments (Single-task)
        if getattr(train_args, 'task_adaptive_encoder', False):
            encoder = TaskAdaptiveEncoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout, num_blocks=encoder_num_blocks)
        else:
            encoder = Encoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout, num_blocks=encoder_num_blocks)

        if task_name == 'rf_fingerprinting':
            task_head = RFClassificationHead(input_dim=2*train_args.d2, num_classes=num_classes, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)
        elif task_name == 'channel_estimation':
            task_head = ChannelEstimationHead(input_dim=2*train_args.d2, hidden_dim=train_args.head_hidden_dim, output_length=52, dropout=train_args.dropout)
        elif task_name == 'cfo_estimation':
            # Choose CFO head based on training arguments
            if getattr(train_args, 'direct_cfo', False):
                task_head = DirectCFOEstimationHead(input_seq_len=train_args.proj_seq_len, hidden_dim=128, dropout=0.1)
            elif getattr(train_args, 'adaptive_cfo', False):
                task_head = CFOAdaptiveHead(input_dim=2*train_args.d2, hidden_dim=128, dropout=0.1)
            elif getattr(train_args, 'simple_cfo', False):
                task_head = SimpleCFOEstimationHead(input_dim=2*train_args.d2, hidden_dim=64, dropout=0.1)
            else:
                task_head = CFOEstimationHead(input_dim=2*train_args.d2, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)
        else:
            raise ValueError(f"Unknown task: {task_name}")

        model = torch.nn.ModuleDict({
            'projection': projection,
            'encoder': encoder,
            'head': task_head
        })

    model.to(device)

    # Load trained weights
    print(f"Loading weights from {cli_args.model_path}")
    checkpoint = torch.load(cli_args.model_path, map_location=device)
    
    if is_mtl:
        model['projections'].load_state_dict(checkpoint['projections_state_dict'])
        model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
        model['heads'].load_state_dict(checkpoint['heads_state_dict'])
    else:
        # Single-task models saved as a single state dict
        if 'model_state_dict' in checkpoint: 
            model_state_dict = checkpoint['model_state_dict']
            
            # For CFO estimation with UpsamplingProjector, filter out projection keys since it has no parameters
            # The new models (both single and MTL) are saved with a consistent 
            # ModuleDict structure, so we can load the state dict directly.
            # The old filtering logic for legacy models is no longer needed.
            model.load_state_dict(model_state_dict)
        
        # Format for models where each module is saved separately
        elif 'projection_state_dict' in checkpoint and 'encoder_state_dict' in checkpoint and 'head_state_dict' in checkpoint:
            model['projection'].load_state_dict(checkpoint['projection_state_dict'])
            model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
            model['head'].load_state_dict(checkpoint['head_state_dict'])
        
        else:
            raise KeyError("Could not find model weights in a recognized format in the checkpoint.")

    model.eval()
    print("Model loaded successfully.")
    
    # Determine tasks to evaluate
    tasks_to_evaluate = train_args.task if is_mtl else [train_args.task]
    print(f"Found tasks: {tasks_to_evaluate}. Running evaluation for each.")

    # Evaluation loop for all relevant tasks
    for task in tasks_to_evaluate:
        print(f"\n===== Evaluating Task: {task} =====")

        # Task head validation
        if is_mtl:
            task_head = model['heads'][task]
        else:
            task_head = model['head']

        task_valid = False
        if task == 'rf_fingerprinting' and isinstance(task_head, RFClassificationHead):
            task_valid = True
        elif task == 'cfo_estimation' and isinstance(task_head, (CFOEstimationHead, SimpleCFOEstimationHead, DirectCFOEstimationHead, CFOAdaptiveHead)):
            task_valid = True
        elif task == 'channel_estimation' and isinstance(task_head, ChannelEstimationHead):
            task_valid = True

        if not task_valid:
            print(f"Warning: Mismatched model head for task '{task}'. Head is {type(task_head).__name__}. Skipping.")
            continue
        else:
            print(f"Using {type(task_head).__name__} for task '{task}'.")

        if task == 'rf_fingerprinting':
            evaluate_rf_fingerprinting(model, test_dl, device, model_dir, list(ID_class_dict.keys()), train_args)
        elif task == 'cfo_estimation':
            evaluate_cfo_estimation(model, test_dl, device, model_dir, max_cfo, mean_cfo, std_cfo, train_args)
        elif task == 'channel_estimation':
            evaluate_channel_estimation(model, test_dl, device, model_dir, train_args)

if __name__ == '__main__':
    main() 
