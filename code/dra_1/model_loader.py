import os
import json
import argparse
import torch
import sys

# Add parent directories to path to import modules from other folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rep_lr')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rep_lr.models import (
    ComplexSequenceProjector, UpsamplingProjector, Encoder, TaskAdaptiveEncoder,
    RFClassificationHead, ChannelEstimationHead, CFOEstimationHead,
    SimpleCFOEstimationHead, DirectCFOEstimationHead, CFOAdaptiveHead
)


def find_best_checkpoint(experiment_path):
    """
    Find the best checkpoint (.pt file) in an experiment directory.
    Prioritizes files with 'best' in the name.
    
    Args:
        experiment_path (str): Path to experiment directory
        
    Returns:
        str: Path to the best checkpoint file
    """
    if not os.path.isdir(experiment_path):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")
    
    model_path = None
    for f in sorted(os.listdir(experiment_path), reverse=True):
        if f.endswith(".pt"):
            if "best" in f:  # Prioritize "best.pt"
                model_path = os.path.join(experiment_path, f)
                break
            if model_path is None:
                model_path = os.path.join(experiment_path, f)
    
    if not model_path:
        raise FileNotFoundError(f"Could not find a .pt model file in {experiment_path}")
    
    return model_path


def load_training_args(experiment_path):
    """
    Load training arguments from args.json in experiment directory.
    
    Args:
        experiment_path (str): Path to experiment directory
        
    Returns:
        argparse.Namespace: Training arguments
    """
    args_path = os.path.join(experiment_path, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {experiment_path}")
    
    with open(args_path, 'r') as f:
        train_args = argparse.Namespace(**json.load(f))
    
    return train_args


def create_projection_layer(task, train_args):
    """
    Create a projection layer for a specific task based on training arguments.
    
    Args:
        task (str): Task name ('rf_fingerprinting', 'cfo_estimation', 'channel_estimation')
        train_args (argparse.Namespace): Training arguments
        
    Returns:
        torch.nn.Module: Projection layer
    """
    if task == 'rf_fingerprinting':
        seq_len = train_args.slice_len
        return ComplexSequenceProjector(
            input_seq_len=seq_len, 
            output_seq_len=train_args.proj_seq_len, 
            hidden_dim=train_args.proj_hidden_dim
        )
    elif task == 'channel_estimation':
        seq_len = 160
        return ComplexSequenceProjector(
            input_seq_len=seq_len, 
            output_seq_len=train_args.proj_seq_len, 
            hidden_dim=train_args.proj_hidden_dim
        )
    elif task == 'cfo_estimation':
        # In the updated training code, CFO estimation now uses ComplexSequenceProjector
        # by default for better feature extraction, unless explicitly using UpsamplingProjector
        seq_len = 160
        return ComplexSequenceProjector(
            input_seq_len=seq_len, 
            output_seq_len=train_args.proj_seq_len, 
            hidden_dim=train_args.proj_hidden_dim
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def create_encoder(train_args):
    """
    Create encoder based on training arguments.
    
    Args:
        train_args (argparse.Namespace): Training arguments
        
    Returns:
        torch.nn.Module: Encoder
    """
    if getattr(train_args, 'task_adaptive_encoder', False):
        return TaskAdaptiveEncoder(
            slice_size=train_args.proj_seq_len,
            output_dim=train_args.d2,
            dropout=train_args.dropout,
            num_blocks=getattr(train_args, 'encoder_num_blocks', 3)
        )
    else:
        return Encoder(
            slice_size=train_args.proj_seq_len,
            output_dim=train_args.d2,
            dropout=train_args.dropout,
            num_blocks=getattr(train_args, 'encoder_num_blocks', 1)
        )


def create_task_head(task, train_args, num_classes=16):
    """
    Create a task head based on training arguments.
    
    Args:
        task (str): Task name
        train_args (argparse.Namespace): Training arguments
        num_classes (int): Number of classes for classification tasks
        
    Returns:
        torch.nn.Module: Task head
    """
    if task == 'rf_fingerprinting':
        return RFClassificationHead(
            input_dim=2*train_args.d2,
            num_classes=num_classes,
            hidden_dim=train_args.head_hidden_dim,
            dropout=train_args.dropout
        )
    elif task == 'channel_estimation':
        return ChannelEstimationHead(
            input_dim=2*train_args.d2,
            hidden_dim=train_args.head_hidden_dim,
            output_length=52,
            dropout=train_args.dropout
        )
    elif task == 'cfo_estimation':
        # Handle different CFO head types
        if getattr(train_args, 'adaptive_cfo', False):
            return CFOAdaptiveHead(
                input_dim=2*train_args.d2,
                hidden_dim=128,
                dropout=0.1
            )
        elif getattr(train_args, 'simple_cfo', False):
            return SimpleCFOEstimationHead(
                input_dim=2*train_args.d2,
                hidden_dim=64,
                dropout=0.1
            )
        elif getattr(train_args, 'direct_cfo', False):
            return DirectCFOEstimationHead(
                input_seq_len=train_args.proj_seq_len,
                hidden_dim=128,
                dropout=0.1
            )
        else:
            return CFOEstimationHead(
                input_dim=2*train_args.d2,
                hidden_dim=train_args.head_hidden_dim,
                dropout=train_args.dropout
            )
    else:
        raise ValueError(f"Unknown task: {task}")


def load_model_from_experiment(experiment_path, device=None, load_heads=False):
    """
    Load a complete model from an experiment directory.
    
    Args:
        experiment_path (str): Path to experiment directory containing args.json and checkpoint
        device (torch.device, optional): Device to load model on
        load_heads (bool): Whether to load task heads
        
    Returns:
        dict: Dictionary containing:
            - 'model': The loaded model (ModuleDict)
            - 'train_args': Training arguments
            - 'checkpoint': Raw checkpoint data
            - 'is_mtl': Whether this is an MTL model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load training arguments
    train_args = load_training_args(experiment_path)
    
    # Find and load checkpoint
    model_path = find_best_checkpoint(experiment_path)
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine if this is MTL or single-task
    is_mtl = getattr(train_args, 'mtl', False)
    
    if is_mtl:
        return _load_mtl_model(train_args, checkpoint, device, load_heads)
    else:
        return _load_single_task_model(train_args, checkpoint, device, load_heads)


def _load_mtl_model(train_args, checkpoint, device, load_heads):
    """Load MTL model components."""
    print("Loading MTL model architecture...")
    
    # Create projections
    projections = torch.nn.ModuleDict()
    for task in train_args.task:
        projections[task] = create_projection_layer(task, train_args)
    
    # Create encoder
    encoder = create_encoder(train_args)
    
    # Create model dict
    model = torch.nn.ModuleDict({
        'projections': projections,
        'encoder': encoder
    })
    
    # Load weights
    model['projections'].load_state_dict(checkpoint['projections_state_dict'])
    model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
    
    # Load heads if requested
    if load_heads:
        heads = torch.nn.ModuleDict()
        for task in train_args.task:
            heads[task] = create_task_head(task, train_args)
        heads.load_state_dict(checkpoint['heads_state_dict'])
        model['heads'] = heads
    
    model.to(device).eval()
    
    return {
        'model': model,
        'train_args': train_args,
        'checkpoint': checkpoint,
        'is_mtl': True
    }


def _load_single_task_model(train_args, checkpoint, device, load_heads):
    """Load single-task model components."""
    print("Loading single-task model architecture...")
    
    task = train_args.task
    
    # Create components
    projection = create_projection_layer(task, train_args)
    encoder = create_encoder(train_args)
    
    # Create model dict and add components
    model = torch.nn.ModuleDict({
        'projection': projection,
        'encoder': encoder
    })
    
    # Add head if requested, so the model structure matches the checkpoint
    if load_heads:
        head = create_task_head(task, train_args)
        model['head'] = head
    
    # Load weights - handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # New format: complete model state dict.
        # If we aren't loading the head, load non-strictly to ignore head keys.
        model.load_state_dict(checkpoint['model_state_dict'], strict=load_heads)
    else:
        # Old format: separate module states
        if 'module_0' in checkpoint:
            model['projection'].load_state_dict(checkpoint['module_0'])
        if 'module_1' in checkpoint:
            model['encoder'].load_state_dict(checkpoint['module_1'])
        else:
            raise KeyError("Could not find model weights in a recognized legacy format in the checkpoint.")
        
        if load_heads:
            if 'module_2' in checkpoint:
                model['head'].load_state_dict(checkpoint['module_2'])
            else:
                print("Warning: Could not find head weights in legacy checkpoint.")

    model.to(device).eval()
    
    return {
        'model': model,
        'train_args': train_args,
        'checkpoint': checkpoint,
        'is_mtl': False
    }


def load_model_for_extraction(experiment_path, device=None):
    """
    Convenience function to load model for activation extraction.
    Only loads projection and encoder components.
    """
    return load_model_from_experiment(experiment_path, device, load_heads=False)


def load_model_for_attack(experiment_path, device=None):
    """
    Convenience function to load model for data reconstruction attacks.
    Only loads projection and encoder components.
    """
    return load_model_from_experiment(experiment_path, device, load_heads=False)


def load_model_for_utility_check(experiment_path, device=None):
    """
    Convenience function to load model for utility checking.
    Loads all components including task heads.
    """
    return load_model_from_experiment(experiment_path, device, load_heads=True) 