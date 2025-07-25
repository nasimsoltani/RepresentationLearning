#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict



def extract_distance_from_filename(filename):
    """Extract distance in feet from filename using regex pattern."""
    match = re.search(r'_(\d+)ft_', filename)
    if match:
        return int(match.group(1))
    return None

def load_predictions(predictions_path):
    """Load predictions from JSON file."""
    try:
        with open(predictions_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Predictions file not found at {predictions_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {predictions_path}")
        return None

def calculate_cfo_metrics_by_distance(predictions):
    """Calculate CFO MSE metrics grouped by distance."""
    distance_data = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    
    for item in predictions:
        filename = item['filename']
        distance = extract_distance_from_filename(filename)
        if distance is not None:
            # Use real values (denormalized)
            y_true = item['y_true_real']
            y_pred = item['y_pred_real']
            distance_data[distance]['y_true'].append(y_true)
            distance_data[distance]['y_pred'].append(y_pred)
    
    # Calculate MSE for each distance
    distance_mse = {}
    for distance, data in distance_data.items():
        if len(data['y_true']) > 0:
            y_true = np.array(data['y_true'])
            y_pred = np.array(data['y_pred'])
            mse = np.mean((y_true - y_pred) ** 2)
            distance_mse[distance] = mse
    
    return distance_mse

def calculate_channel_metrics_by_distance(predictions):
    """Calculate Channel MSE metrics grouped by distance."""
    distance_data = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    
    for item in predictions:
        filename = item['filename']
        distance = extract_distance_from_filename(filename)
        if distance is not None:
            # Channel predictions are stored as [real_part, imag_part, subcarriers]
            y_true = np.array(item['y_true'])  # [2, 52] format
            y_pred = np.array(item['y_pred'])  # [2, 52] format
            distance_data[distance]['y_true'].append(y_true)
            distance_data[distance]['y_pred'].append(y_pred)
    
    # Calculate MSE for each distance
    distance_mse = {}
    for distance, data in distance_data.items():
        if len(data['y_true']) > 0:
            y_true = np.array(data['y_true'])  # [num_samples, 2, 52]
            y_pred = np.array(data['y_pred'])  # [num_samples, 2, 52]
            
            # Convert to complex numbers for more meaningful MSE
            y_true_complex = y_true[:, 0, :] + 1j * y_true[:, 1, :]
            y_pred_complex = y_pred[:, 0, :] + 1j * y_pred[:, 1, :]
            
            # Calculate MSE
            mse = np.mean(np.abs(y_true_complex - y_pred_complex) ** 2)
            distance_mse[distance] = mse
    
    return distance_mse

def calculate_rf_metrics_by_distance(predictions):
    """Calculate RF accuracy metrics grouped by distance."""
    distance_data = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for item in predictions:
        filename = item['filename']
        distance = extract_distance_from_filename(filename)
        if distance is not None:
            pred_class = item['pred_class']
            gt_class = item['gt_class']
            
            distance_data[distance]['total'] += 1
            if pred_class == gt_class:
                distance_data[distance]['correct'] += 1
    
    # Calculate accuracy for each distance
    distance_accuracy = {}
    for distance, data in distance_data.items():
        if data['total'] > 0:
            accuracy = data['correct'] / data['total']
            distance_accuracy[distance] = accuracy
    
    return distance_accuracy

def find_prediction_file(base_path, filename):
    """
    Search for a specific file within a directory tree.
    Returns the full path to the first match found, or None if not found.
    """
    if not os.path.isdir(base_path):
        # This case is handled in main, but good practice to check here.
        return None
        
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    
    # If the loop completes without finding the file
    return None

def plot_cfo_comparison(experiment_path, output_dir):
    """Generate CFO MSE vs Distance comparison plot."""
    # Find prediction files in their respective subdirectories
    cfo_single_path = find_prediction_file(os.path.join(experiment_path, 'cfo'), 'cfo_predictions.json')
    cfo_joint_path = find_prediction_file(os.path.join(experiment_path, 'rf_cfo_channel'), 'cfo_predictions.json')

    if not cfo_single_path:
        print("Warning: Could not find 'cfo_predictions.json' in cfo/ subdirectory.")
    if not cfo_joint_path:
        print("Warning: Could not find 'cfo_predictions.json' in rf_cfo_channel/ subdirectory.")
    if not cfo_single_path or not cfo_joint_path:
        return

    cfo_single_predictions = load_predictions(cfo_single_path)
    cfo_joint_predictions = load_predictions(cfo_joint_path)
    
    if cfo_single_predictions is None or cfo_joint_predictions is None:
        print("Warning: Could not load CFO predictions for comparison")
        return
    
    # Calculate metrics
    single_mse = calculate_cfo_metrics_by_distance(cfo_single_predictions)
    joint_mse = calculate_cfo_metrics_by_distance(cfo_joint_predictions)
    
    # Get common distances and sort
    common_distances = sorted(set(single_mse.keys()) & set(joint_mse.keys()))
    
    if not common_distances:
        print("Warning: No common distances found for CFO comparison")
        return
    
    single_mse_values = [single_mse[d] for d in common_distances]
    joint_mse_values = [joint_mse[d] for d in common_distances]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(common_distances, single_mse_values, 'b-o', label='CFO Only (Single-task)', linewidth=2, markersize=6)
    plt.plot(common_distances, joint_mse_values, 'r-s', label='CFO + RF + Channel (Multi-task)', linewidth=2, markersize=6)
    
    plt.xlabel('Distance (ft)', fontsize=12)
    plt.ylabel('MSE (Hz²)', fontsize=12)
    plt.title('CFO Estimation: MSE vs Distance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'cfo_mse_vs_distance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CFO MSE vs Distance plot saved to {plot_path}")

def plot_channel_comparison(experiment_path, output_dir):
    """Generate Channel MSE vs Distance comparison plot."""
    # Find prediction files in their respective subdirectories
    channel_single_path = find_prediction_file(os.path.join(experiment_path, 'channel'), 'channel_predictions.json')
    channel_joint_path = find_prediction_file(os.path.join(experiment_path, 'rf_cfo_channel'), 'channel_predictions.json')

    if not channel_single_path:
        print("Warning: Could not find 'channel_predictions.json' in channel/ subdirectory.")
    if not channel_joint_path:
        print("Warning: Could not find 'channel_predictions.json' in rf_cfo_channel/ subdirectory.")
    if not channel_single_path or not channel_joint_path:
        return

    channel_single_predictions = load_predictions(channel_single_path)
    channel_joint_predictions = load_predictions(channel_joint_path)
    
    if channel_single_predictions is None or channel_joint_predictions is None:
        print("Warning: Could not load Channel predictions for comparison")
        return
    
    # Calculate metrics
    single_mse = calculate_channel_metrics_by_distance(channel_single_predictions)
    joint_mse = calculate_channel_metrics_by_distance(channel_joint_predictions)
    
    # Get common distances and sort
    common_distances = sorted(set(single_mse.keys()) & set(joint_mse.keys()))
    
    if not common_distances:
        print("Warning: No common distances found for Channel comparison")
        return
    
    single_mse_values = [single_mse[d] for d in common_distances]
    joint_mse_values = [joint_mse[d] for d in common_distances]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(common_distances, single_mse_values, 'g-o', label='Channel Only (Single-task)', linewidth=2, markersize=6)
    plt.plot(common_distances, joint_mse_values, 'purple', linestyle='-', marker='s', label='Channel + RF + CFO (Multi-task)', linewidth=2, markersize=6)
    
    plt.xlabel('Distance (ft)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Channel Estimation: MSE vs Distance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'channel_mse_vs_distance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Channel MSE vs Distance plot saved to {plot_path}")

def plot_rf_comparison(experiment_path, output_dir):
    """Generate RF Accuracy vs Distance comparison plot."""
    # Find prediction files in their respective subdirectories
    rf_single_path = find_prediction_file(os.path.join(experiment_path, 'rf'), 'predictions.json')
    rf_joint_path = find_prediction_file(os.path.join(experiment_path, 'rf_cfo_channel'), 'predictions.json')

    if not rf_single_path:
        print("Warning: Could not find 'predictions.json' in rf/ subdirectory.")
    if not rf_joint_path:
        print("Warning: Could not find 'predictions.json' in rf_cfo_channel/ subdirectory.")
    if not rf_single_path or not rf_joint_path:
        return

    rf_single_predictions = load_predictions(rf_single_path)
    rf_joint_predictions = load_predictions(rf_joint_path)
    
    if rf_single_predictions is None or rf_joint_predictions is None:
        print("Warning: Could not load RF predictions for comparison")
        return
    
    # Calculate metrics
    single_accuracy = calculate_rf_metrics_by_distance(rf_single_predictions)
    joint_accuracy = calculate_rf_metrics_by_distance(rf_joint_predictions)
    
    # Get common distances and sort
    common_distances = sorted(set(single_accuracy.keys()) & set(joint_accuracy.keys()))
    
    if not common_distances:
        print("Warning: No common distances found for RF comparison")
        return
    
    single_accuracy_values = [single_accuracy[d] for d in common_distances]
    joint_accuracy_values = [joint_accuracy[d] for d in common_distances]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(common_distances, single_accuracy_values, 'orange', linestyle='-', marker='o', label='RF Only (Single-task)', linewidth=2, markersize=6)
    plt.plot(common_distances, joint_accuracy_values, 'teal', linestyle='-', marker='s', label='RF + CFO + Channel (Multi-task)', linewidth=2, markersize=6)
    
    plt.xlabel('Distance (ft)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('RF Fingerprinting: Accuracy vs Distance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)  # Accuracy should be between 0 and 1
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'rf_accuracy_vs_distance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RF Accuracy vs Distance plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate line plots comparing single-task vs multi-task training performance.')
    parser.add_argument('--experiment_path', type=str, required=True, 
                        help='Path to the experiment directory containing cfo, channel, rf, and rf_cfo_channel subdirectories.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the generated plots. Defaults to experiment_path/plots.')
    
    args = parser.parse_args()
    
    # Validate experiment path
    if not os.path.exists(args.experiment_path):
        raise FileNotFoundError(f"Experiment path not found: {args.experiment_path}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_path, 'plots')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving plots to: {args.output_dir}")
    
    # Check for required subdirectories
    required_dirs = ['cfo', 'channel', 'rf', 'rf_cfo_channel']
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(args.experiment_path, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"Warning: Missing required subdirectories: {missing_dirs}")
        print("Available subdirectories:")
        for item in os.listdir(args.experiment_path):
            item_path = os.path.join(args.experiment_path, item)
            if os.path.isdir(item_path):
                print(f"  - {item}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    
    print("\n1. Generating CFO MSE vs Distance plot...")
    plot_cfo_comparison(args.experiment_path, args.output_dir)
    
    print("\n2. Generating Channel MSE vs Distance plot...")
    plot_channel_comparison(args.experiment_path, args.output_dir)
    
    print("\n3. Generating RF Accuracy vs Distance plot...")
    plot_rf_comparison(args.experiment_path, args.output_dir)
    
    print(f"\nAll plots generated successfully in {args.output_dir}")

if __name__ == '__main__':
    main()