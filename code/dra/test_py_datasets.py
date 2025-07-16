import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Add the 'code' directory to sys.path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra.py_datasets import ActivationDataset

def test_dataset(args):
    """
    Tests the ActivationDataset by loading it and fetching a few samples.
    """
    print(f"--- Testing with noise_type: {args.noise_type}, noise_level: {args.noise_level} ---")

    try:
        # Initialize the dataset
        dataset = ActivationDataset(args.activations_dir, args)
        
        # Check if the dataset is empty
        if len(dataset) == 0:
            print("Dataset is empty. Cannot perform test.")
            return

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Get one batch of data
        sample_batch = next(iter(dataloader))

        # Print shapes and types to verify
        print(f"Successfully loaded a batch of size {args.batch_size}.")
        original_activation = sample_batch['activation']
        print(f"  Activation shape: {original_activation.shape}")
        print(f"  Activation dtype: {original_activation.dtype}")

        # Check a few other keys
        print(f"  RF_X shape: {sample_batch['RF_X'].shape}")
        print(f"  Channel_X shape: {sample_batch['Channel_X'].shape}")
        print(f"  Filename of first sample: {sample_batch['filename'][0]}")

        # If noise was applied, compare original vs noisy activation
        # Note: To do a perfect comparison, we'd need to load a sample without noise first.
        # This basic test just ensures the noisy activation is loaded correctly.
        if args.noise_type != 'none':
            print("  Noise has been applied to the activations.")
            # Simple check: Mean of activation should be different from a no-noise run
            print(f"  Mean of (potentially noisy) activation: {torch.mean(original_activation).item():.6f}")

        print("--- Test PASSED ---\n")

    except Exception as e:
        print(f"--- Test FAILED with an error: {e} ---")
        import traceback
        traceback.print_exc()
        print("\n")


if __name__ == '__main__':
    # --- Base Arguments ---
    base_args = argparse.Namespace(
        activations_dir='/home/hofmann/Documents/projects/RepresentationLearning/dataset/activations/train',
        model_path='/home/hofmann/Documents/projects/RepresentationLearning/results_20250715_202527/rf_cfo_channel/rf_fingerprinting_cfo_estimation_channel_estimation_20250715_202529/rf_fingerprinting_cfo_estimation_channel_estimation_best.pt',
        gpu_id=0,
        batch_size=4
    )

    # --- Test Case 1: No Noise ---
    args_no_noise = argparse.Namespace(**vars(base_args))
    args_no_noise.noise_type = 'none'
    args_no_noise.noise_level = 0.0
    test_dataset(args_no_noise)

    # --- Test Case 2: Isotropic Noise ---
    args_iso_noise = argparse.Namespace(**vars(base_args))
    args_iso_noise.noise_type = 'isotropic'
    args_iso_noise.noise_level = 0.1
    test_dataset(args_iso_noise)

    # --- Test Case 3: Anisotropic Noise ---
    args_aniso_noise = argparse.Namespace(**vars(base_args))
    args_aniso_noise.noise_type = 'anisotropic'
    args_aniso_noise.noise_level = 0.1
    test_dataset(args_aniso_noise) 