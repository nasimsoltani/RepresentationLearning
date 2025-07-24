#!/usr/bin/env python3
"""
Example script to run the reconstruction attack.

Usage:
    python run_attack_example.py
"""

import os
import subprocess
import argparse

def run_attack(experiment_path, output_dir="./attack_results", epochs=50, batch_size=32, patience=10, plot_interval=5, noise_type='isotropic', fim_path=None):
    """
    Run the reconstruction attack with specified parameters.
    
    Args:
        experiment_path (str): Path to experiment directory containing activations
        output_dir (str): Output directory for attack results
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        patience (int): Early stopping patience
        plot_interval (int): Plot reconstructions every N epochs
        noise_type (str): Type of noise ('isotropic' or 'nonisotropic')
        fim_path (str): Path to FIM eigendecomposition file (required for nonisotropic noise)
    """
    
    cmd = [
        "python", "new_attack.py",
        "--experiment_path", experiment_path,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--patience", str(patience),
        "--learning_rate", "1e-4",
        "--noise_level", "0.0",  # Start with no noise
        "--noise_type", noise_type,
        "--plot_interval", str(plot_interval),
    ]
    
    # Add FIM path if provided for non-isotropic noise
    if noise_type == 'nonisotropic' and fim_path:
        cmd.extend(["--fim_path", fim_path])
    
    print(f"Running attack command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Attack completed successfully!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Attack failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description='Run reconstruction attack example')
    parser.add_argument('--experiment_path', type=str, 
                       default='/home/hofmann/Documents/projects/RepresentationLearning/results_20250720_172807/rf_cfo_channel/rf_fingerprinting_cfo_estimation_channel_estimation_20250720_172811',
                       help='Path to experiment directory')
    parser.add_argument('--output_dir', type=str, default='./attack_results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--plot_interval', type=int, default=5,
                       help='Plot reconstructions every N epochs')
    parser.add_argument('--noise_type', type=str, default='isotropic',
                       choices=['isotropic', 'nonisotropic'],
                       help='Type of noise to use for attack')
    parser.add_argument('--fim_path', type=str, default=None,
                       help='Path to FIM eigendecomposition file for non-isotropic noise')
    
    args = parser.parse_args()
    
    # Check if experiment path exists
    if not os.path.exists(args.experiment_path):
        print(f"Error: Experiment path does not exist: {args.experiment_path}")
        print("Please provide a valid experiment path containing an 'activations' folder.")
        return
    
    activations_path = os.path.join(args.experiment_path, 'activations')
    if not os.path.exists(activations_path):
        print(f"Error: Activations folder not found in: {activations_path}")
        print("Make sure the experiment path contains an 'activations' subdirectory.")
        return
    
    print(f"Found activations directory: {activations_path}")
    activation_files = [f for f in os.listdir(activations_path) if f.endswith('.pth')]
    print(f"Found {len(activation_files)} activation files")
    
    if len(activation_files) == 0:
        print("Error: No .pth activation files found!")
        return
    
    # Validate non-isotropic noise requirements
    if args.noise_type == 'nonisotropic' and args.fim_path is None:
        print("Error: --fim_path is required when using non-isotropic noise")
        return
    
    if args.noise_type == 'nonisotropic' and not os.path.exists(args.fim_path):
        print(f"Error: FIM file not found: {args.fim_path}")
        return
    
    # Run the attack
    run_attack(
        experiment_path=args.experiment_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        plot_interval=args.plot_interval,
        noise_type=args.noise_type,
        fim_path=args.fim_path
    )

def run_both_noise_types_example(experiment_path):
    """
    Example function to run both isotropic and non-isotropic attacks for comparison.
    
    Note: This assumes you have a FIM eigendecomposition file available.
    """
    print("=" * 60)
    print("EXAMPLE: Running both noise types for comparison")
    print("=" * 60)
    
    # Example FIM path (update this to your actual FIM file)
    fim_path = "path/to/your/fim_eigendecomposition.pt"
    
    print("\n1. Running attack with ISOTROPIC noise...")
    run_attack(
        experiment_path=experiment_path,
        output_dir="./attack_results_isotropic",
        epochs=30,
        noise_type='isotropic'
    )
    
    if os.path.exists(fim_path):
        print("\n2. Running attack with NON-ISOTROPIC noise...")
        run_attack(
            experiment_path=experiment_path,
            output_dir="./attack_results_nonisotropic", 
            epochs=30,
            noise_type='nonisotropic',
            fim_path=fim_path
        )
    else:
        print(f"\n2. Skipping non-isotropic attack - FIM file not found: {fim_path}")
        print("   To run non-isotropic attacks, provide the path to your FIM eigendecomposition file.")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("Check the output directories to compare results!")
    print("=" * 60)


if __name__ == '__main__':
    main() 