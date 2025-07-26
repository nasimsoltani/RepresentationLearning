"""
Script to diagnose and check for model loading errors in PyTorch.

This script loads a model checkpoint and its corresponding arguments, 
then compares the keys in the saved state_dict with the keys in a newly 
instantiated model. It helps identify mismatches that cause runtime errors.

Usage:
    python code/dra_1/check_model_loader.py --experiment_path /path/to/your/experiment/dir
"""
import argparse
import torch
import json
import os
import sys

# --- Add necessary paths to sys.path ---
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the code directory (parent of dra_1)
code_dir = os.path.dirname(script_dir)
# Add the project root (parent of code)
project_root = os.path.dirname(code_dir)

# Add both to sys.path
sys.path.insert(0, code_dir)
sys.path.insert(0, project_root)

# Now we can import the models module
from rep_lr import models


def get_head_class(task_name):
    """Maps a task name string to its corresponding model class."""
    head_map = {
        'rf': models.RFClassificationHead,
        'cfo': models.CFOEstimationHead,
        'channel': models.ChannelEstimationHead,
    }
    if task_name not in head_map:
        raise ValueError(f"Unknown task name: {task_name}. Available tasks: {list(head_map.keys())}")
    return head_map[task_name]

def create_projection_layer(task_name, train_args):
    """Creates a projection layer based on task and args, mimicking model_loader."""
    proj_seq_len = train_args.get('proj_seq_len')
    proj_hidden_dim = train_args.get('proj_hidden_dim')

    # Determine input sequence length based on task
    if 'rf' in task_name:
        seq_len = train_args.get('slice_len') or train_args.get('slice_size')
    elif 'cfo' in task_name or 'channel' in task_name:
        seq_len = 160 # As per model_loader.py
    else:
        raise ValueError(f"Unknown task for projection: {task_name}")

    if not all([proj_seq_len, proj_hidden_dim, seq_len]):
        raise KeyError("Missing one of 'proj_seq_len', 'proj_hidden_dim', or 'slice_len' in args.json")

    # In your training code, it appears you are using ComplexSequenceProjector
    return models.ComplexSequenceProjector(
        input_seq_len=seq_len,
        output_seq_len=proj_seq_len,
        hidden_dim=proj_hidden_dim
    )

def check_state_dict_keys(model_state_dict, checkpoint_state_dict, component_name):
    """
    Compares keys between a model's state_dict and a checkpoint's state_dict
    and prints a detailed report.
    """
    print(f"\n--- Checking Keys for: {component_name} ---")
    
    model_keys = set(model_state_dict.keys())
    ckpt_keys = set(checkpoint_state_dict.keys())

    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys

    if not missing_keys and not unexpected_keys:
        print(f"✅ SUCCESS: All keys match for {component_name}.")
        return True
    
    print(f"❌ MISMATCH: Found issues with keys for {component_name}.")
    if missing_keys:
        print("\n[+] Missing Keys (expected by the model, but not in the checkpoint file):")
        for key in sorted(list(missing_keys)):
            print(f"  - {key}")

    if unexpected_keys:
        print("\n[+] Unexpected Keys (in the checkpoint file, but not expected by the model):")
        for key in sorted(list(unexpected_keys)):
            print(f"  - {key}")
    
    return False

def main(args):
    """Main function to run the model checker."""
    if not os.path.isdir(args.experiment_path):
        print(f"Error: Experiment path not found at '{args.experiment_path}'")
        return

    # --- 1. Load configuration and checkpoint ---
    args_path = os.path.join(args.experiment_path, 'args.json')
    
    if not os.path.exists(args_path):
        print(f"Error: Could not find 'args.json' in '{args.experiment_path}'")
        return

    # Look for checkpoint files with different naming patterns
    checkpoint_path = None
    possible_names = [
        'model.pth',
        'model.pt', 
        'best_model.pt',
        'best_model.pth'
    ]
    
    # Also look for task-specific checkpoint files
    with open(args_path, 'r') as f:
        train_args = json.load(f)
    
    # Handle different ways task name might be stored
    task_name = train_args.get('task_name') or train_args.get('task', 'unknown')
    # Extract the base task name (e.g., 'channel' from 'channel_estimation')
    if '_' in task_name:
        base_task = task_name.split('_')[0]
    else:
        base_task = task_name
    
    print(f"Detected task: {task_name}, base task: {base_task}")
    
    task_specific_names = [
        f'{base_task}_best.pt',
        f'{base_task}_best.pth',
        f'{base_task}_final.pt',
        f'{base_task}_final.pth',
        f'{task_name}_best.pt',
        f'{task_name}_best.pth',
        f'{task_name}_final.pt',
        f'{task_name}_final.pth',
    ]
    
    # Also look for files that match common patterns we see in the directory
    # Get all .pt/.pth files and check for ones that contain "best"
    all_checkpoint_files = [f for f in os.listdir(args.experiment_path) 
                           if f.endswith(('.pt', '.pth'))]
    
    # Look for any file with "best" in the name
    best_files = [f for f in all_checkpoint_files if 'best' in f.lower()]
    
    if best_files:
        task_specific_names.extend(best_files)
    
    # Check all possible names
    for name in possible_names + task_specific_names:
        temp_path = os.path.join(args.experiment_path, name)
        if os.path.exists(temp_path):
            checkpoint_path = temp_path
            break
    
    if checkpoint_path is None:
        print(f"Error: Could not find any checkpoint file in '{args.experiment_path}'")
        print("Looked for files with names:")
        for name in possible_names + task_specific_names:
            print(f"  - {name}")
        print("\nAvailable files in directory:")
        for file in os.listdir(args.experiment_path):
            if file.endswith(('.pt', '.pth')):
                print(f"  - {file}")
        return

    print(f"Found checkpoint file: {os.path.basename(checkpoint_path)}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Successfully loaded checkpoint and args for task: '{task_name}'")

    # --- 2. Recreate the full model structure as it was during training ---
    # Handle different key names from older or different args.json files
    slice_size = train_args.get('slice_size') or train_args.get('slice_len')
    proj_seq_len = train_args.get('proj_seq_len')
    if not slice_size or not proj_seq_len:
        raise KeyError("Could not find 'slice_len'/'slice_size' or 'proj_seq_len' in args.json")

    num_blocks = train_args.get('num_blocks') or train_args.get('encoder_num_blocks', 5)

    # Create all three components
    print("INFO: Recreating model components (projection, encoder, head)...")
    projection = create_projection_layer(task_name, train_args)

    if train_args.get('task_adaptive_encoder', False):
        print("INFO: Instantiating TaskAdaptiveEncoder.")
        encoder = models.TaskAdaptiveEncoder(
            slice_size=proj_seq_len, # TaskAdaptiveEncoder works on projected sequence length
            output_dim=train_args['d2'],
            num_blocks=num_blocks
        )
    else:
        print("INFO: Instantiating standard Encoder.")
        encoder = models.Encoder(
            slice_size=proj_seq_len, # Encoder works on projected sequence length
            output_dim=train_args['d2'],
            num_blocks=num_blocks
        )

    HeadClass = get_head_class(base_task)
    if base_task == 'rf':
        num_classes = train_args.get('num_classes')
        if num_classes is None:
            raise KeyError("Could not find 'num_classes' in args.json for RF task.")
        head = HeadClass(input_dim=2 * train_args['d2'], num_classes=num_classes)
    elif base_task == 'channel':
         head = HeadClass(input_dim=2 * train_args['d2'], output_length=train_args.get('output_length', 52))
    else: # cfo
        head = HeadClass(input_dim=2 * train_args['d2'])
    
    # Assemble into a single ModuleDict to match the saved structure
    recreated_full_model = torch.nn.ModuleDict({
        'projection': projection,
        'encoder': encoder,
        'head': head
    })
    
    # --- 3. Compare State Dictionaries ---
    # The checkpoint saves the full model under 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['model_state_dict']
        check_state_dict_keys(
            model_state_dict=recreated_full_model.state_dict(),
            checkpoint_state_dict=checkpoint_state_dict,
            component_name="Full Recreated Model"
        )
    else:
        print("Error: Could not find 'model_state_dict' in the checkpoint.")
        print(f"Available keys in checkpoint: {list(checkpoint.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if a saved PyTorch model checkpoint can be loaded correctly.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--experiment_path',
        type=str,
        required=True,
        help="Path to the experiment directory containing 'model.pth' and 'args.json'."
    )
    args = parser.parse_args()
    main(args) 