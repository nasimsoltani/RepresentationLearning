#!/usr/bin/env python3
"""
Test script to compare barebones.py vs main pipeline with identical settings.
This helps identify why barebones works better than the main pipeline.
"""

import os
import subprocess
import json
from datetime import datetime

def run_barebones_test():
    """Run barebones with specific settings"""
    print("=" * 60)
    print("Running Barebones CFO Estimation")
    print("=" * 60)
    
    cmd = [
        "python", "barebones.py",
        "--pkl_dataset_path", "/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl",
        "--epochs", "10",
        "--batch_size", "64", 
        "--lr", "1e-3",
        "--proj_seq_len", "256",
        "--proj_hidden_dim", "512",
        "--d2", "128",
        "--head_hidden_dim", "256",
        "--dropout", "0.1",
        "--gpu_id", "0",
        "--patience", "5"
    ]
    
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout[-2000:])  # Last 2000 chars
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
    print("Return code:", result.returncode)
    
    return result.returncode == 0

def run_main_pipeline_test():
    """Run main pipeline with barebones-like settings"""
    print("\n" + "=" * 60)
    print("Running Main Pipeline CFO Estimation")
    print("=" * 60)
    
    cmd = [
        "python", "code/rep_lr/main.py",
        "--task", "cfo_estimation",
        "--pkl_dataset_path", "/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl",
        "--save_path", "results_comparison",
        "--epochs", "10",
        "--batch_size", "64",  # Same as barebones
        "--lr", "1e-3",        # Same as barebones (NOT 3e-5!)
        "--proj_seq_len", "256",
        "--proj_hidden_dim", "512", 
        "--d2", "128",
        "--head_hidden_dim", "256",
        "--dropout", "0.1",
        "--encoder_num_blocks", "1",
        "--task_adaptive_encoder",
        "--gpu_id", "0",
        "--patience", "5",
        "--wandb_project", "comparison_test"
    ]
    
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout[-2000:])  # Last 2000 chars  
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
    print("Return code:", result.returncode)
    
    return result.returncode == 0

def compare_results():
    """Compare results from both approaches"""
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)
    
    print("Key differences identified:")
    print("1. Learning Rate: Your command used 3e-5, barebones uses 1e-3 (33x higher!)")
    print("2. Batch Size: Your command used 128, barebones uses 64")
    print("3. WandB: Main pipeline has wandb overhead")
    print("4. Training Loop: Main pipeline more complex")
    
    print("\nRecommended fixes for main pipeline:")
    print("1. Use lr=1e-3 instead of 3e-5")
    print("2. Consider batch_size=64")
    print("3. Optionally disable wandb for debugging")
    
    # Check if results exist
    barebones_results = "results_barebones"
    main_results = "results_comparison"
    
    if os.path.exists(barebones_results):
        print(f"\n✓ Barebones results found in: {barebones_results}")
        # Find latest run
        runs = [d for d in os.listdir(barebones_results) if d.startswith("barebones_cfo_")]
        if runs:
            latest_run = sorted(runs)[-1]
            metrics_file = os.path.join(barebones_results, latest_run, "cfo_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    print(f"  - R² Score: {metrics.get('r2_score', 'N/A'):.4f}")
                    print(f"  - MAE: {metrics.get('mae', 'N/A'):.4f}")
    else:
        print(f"✗ No barebones results found")
    
    if os.path.exists(main_results):
        print(f"\n✓ Main pipeline results found in: {main_results}")
        # Find latest run  
        runs = [d for d in os.listdir(main_results) if d.startswith("cfo_estimation_")]
        if runs:
            latest_run = sorted(runs)[-1]
            metrics_file = os.path.join(main_results, latest_run, "cfo_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    print(f"  - R² Score: {metrics.get('r2_score', 'N/A'):.4f}")
                    print(f"  - MAE: {metrics.get('mae', 'N/A'):.4f}")
    else:
        print(f"✗ No main pipeline results found")

def main():
    print("CFO Estimation: Barebones vs Main Pipeline Comparison")
    print("=" * 60)
    
    # Run tests
    barebones_success = run_barebones_test()
    main_success = run_main_pipeline_test()
    
    # Compare results
    compare_results()
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Barebones: {'✓ Success' if barebones_success else '✗ Failed'}")
    print(f"Main Pipeline: {'✓ Success' if main_success else '✗ Failed'}")
    
    print(f"\nTo fix your main pipeline command, try:")
    print(f"uv run python code/rep_lr/main.py \\")
    print(f"    --task cfo_estimation \\")
    print(f"    --pkl_dataset_path $DATASET_PATH \\")
    print(f"    --save_path \"${{RESULTS_DIR}}/cfo\" \\")
    print(f"    --epochs 300 \\")
    print(f"    --batch_size 64 \\")  # Changed from 128
    print(f"    --lr 1e-3 \\")        # Changed from 3e-5!
    print(f"    --proj_seq_len $PROJ_SEQ_LEN \\")
    print(f"    --proj_hidden_dim $PROJ_HIDDEN_DIM \\")
    print(f"    --d2 $D2 \\")
    print(f"    --head_hidden_dim $HEAD_HIDDEN_DIM \\")
    print(f"    --gpu_id $GPU_ID \\")
    print(f"    --task_adaptive_encoder \\")
    print(f"    --encoder_num_blocks 1")

if __name__ == "__main__":
    main() 