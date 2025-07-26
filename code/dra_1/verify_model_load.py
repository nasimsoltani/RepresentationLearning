"""
Advanced Model Loading Verifier for PyTorch.

This script diagnoses model loading issues by attempting a "dry run" of the 
loading process. Unlike a simple key comparison, it tries to load the state_dict 
from a checkpoint into a newly instantiated model and reports any mismatches 
(missing or unexpected keys) in a detailed, user-friendly format.

This helps predict if a script like `robust_attack.py` will be able to load 
the model weights successfully.

Usage:
    python code/dra_1/verify_model_load.py --experiment_path /path/to/your/experiment/dir
"""
import argparse
import torch
import json
import os
import sys
from collections import OrderedDict

# --- Add necessary paths to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(code_dir)
sys.path.insert(0, code_dir)
sys.path.insert(0, project_root)

# Now we can import from other modules
from rep_lr import models
from dra_1.model_loader import (
    load_training_args, 
    find_best_checkpoint, 
    create_encoder, 
    create_projection_layer
)

def get_head_class(task_name):
    """Maps a task name string to its corresponding model class."""
    head_map = {
        'rf_fingerprinting': models.RFClassificationHead,
        'cfo_estimation': models.CFOEstimationHead,
        'channel_estimation': models.ChannelEstimationHead,
        # Add other potential base tasks here if necessary
        'rf': models.RFClassificationHead,
        'cfo': models.CFOEstimationHead,
        'channel': models.ChannelEstimationHead,
    }
    # Find the correct key by checking if the task_name contains a key
    for key, value in head_map.items():
        if key in task_name:
            return value
    raise ValueError(f"Unknown task name: {task_name}. Available base tasks: {list(head_map.keys())}")

def verify_load(model_component, state_dict, component_name):
    """
    Attempts to load a state_dict into a model component and reports errors.
    """
    print(f"--- Verifying Component: {component_name} ---")
    try:
        # The core of the check: attempt to load the state dict
        incompatible_keys = model_component.load_state_dict(state_dict, strict=False)
        
        missing_keys = incompatible_keys.missing_keys
        unexpected_keys = incompatible_keys.unexpected_keys

        if not missing_keys and not unexpected_keys:
            print(f"✅ SUCCESS: State dictionary for '{component_name}' loaded without any mismatches.")
            return True
        
        print(f"⚠️  WARNING: Found mismatches for '{component_name}'.")
        if missing_keys:
            print("\n[+] Missing Keys (expected by the model, but not in checkpoint):")
            for key in sorted(missing_keys):
                print(f"  - {key}")
        
        if unexpected_keys:
            print("\n[+] Unexpected Keys (in checkpoint, but not expected by the model):")
            for key in sorted(unexpected_keys):
                print(f"  - {key}")
        
        print("\nThis component will likely FAIL to load in other scripts without a compatibility fix.")
        return False

    except Exception as e:
        print(f"❌ CRITICAL ERROR: An exception occurred while trying to load the state_dict for '{component_name}'.")
        print(f"Error: {e}")
        return False

def main(args):
    """Main function to run the model load verifier."""
    try:
        # --- 1. Load Config and Checkpoint ---
        train_args = load_training_args(args.experiment_path)
        checkpoint_path = find_best_checkpoint(args.experiment_path)
        
        print(f"Loaded training args for task: '{train_args.task}'")
        print(f"Found checkpoint: '{os.path.basename(checkpoint_path)}'")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model_state_dict'. Cannot proceed.")

        ckpt_state_dict = checkpoint['model_state_dict']

        # --- 2. Recreate Model Components ---
        print("\nRecreating model architecture from training args...")
        
        is_mtl = getattr(train_args, 'mtl', False)
        
        if is_mtl:
            print("MTL model detected. Verifying encoder and each head.")
            # Verify encoder
            encoder = create_encoder(train_args)
            encoder_state = {k.replace('encoder.', ''): v for k, v in ckpt_state_dict.items() if k.startswith('encoder.')}
            verify_load(encoder, encoder_state, "Encoder")

            # Verify each head
            for task in train_args.task:
                HeadClass = get_head_class(task)
                head = HeadClass(input_dim=2 * train_args.d2) # Simplified instantiation
                head_state = {k.replace(f'heads.{task}.', ''): v for k, v in ckpt_state_dict.items() if k.startswith(f'heads.{task}.')}
                verify_load(head, head_state, f"Head ({task})")

        else: # Single-task model
            print("Single-task model detected. Verifying full model and individual components.")
            task_name = train_args.task
            
            # Recreate each component individually
            projection = create_projection_layer(task_name, train_args)
            encoder = create_encoder(train_args)
            HeadClass = get_head_class(task_name)
            
            # Handle task-specific head arguments
            if 'rf' in task_name:
                head = HeadClass(input_dim=2*train_args.d2, num_classes=train_args.num_classes)
            elif 'channel' in task_name:
                head = HeadClass(input_dim=2*train_args.d2, output_length=getattr(train_args, 'output_length', 52))
            else: # cfo
                head = HeadClass(input_dim=2*train_args.d2)

            # Assemble into a full model to check global load
            full_model = torch.nn.ModuleDict({
                'projection': projection,
                'encoder': encoder,
                'head': head
            })
            
            # --- 3. Perform Verification ---
            # First, check the full model load
            print("\n--- Verifying Full Model Load (Strict) ---")
            try:
                full_model.load_state_dict(ckpt_state_dict, strict=True)
                print("✅ SUCCESS: Full model loaded successfully with strict=True.")
            except RuntimeError as e:
                print("❌ FAILURE: Full model load failed with strict=True. This is the root cause of the error.")
                print("Exact error message:")
                print(f"  {e}")


            # Then, check each component individually using our non-strict method
            print("\n--- Verifying Individual Components (Non-Strict) ---")
            proj_state = OrderedDict((k.replace('projection.', ''), v) for k, v in ckpt_state_dict.items() if k.startswith('projection.'))
            verify_load(projection, proj_state, "Projection")

            encoder_state = OrderedDict((k.replace('encoder.', ''), v) for k, v in ckpt_state_dict.items() if k.startswith('encoder.'))
            verify_load(encoder, encoder_state, "Encoder")

            head_state = OrderedDict((k.replace('head.', ''), v) for k, v in ckpt_state_dict.items() if k.startswith('head.'))
            verify_load(head, head_state, "Head")


    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\n❌ An error occurred during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced script to verify if a PyTorch model checkpoint can be loaded correctly.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--experiment_path',
        type=str,
        required=True,
        help="Path to the experiment directory containing checkpoint and args.json."
    )
    args = parser.parse_args()
    main(args) 