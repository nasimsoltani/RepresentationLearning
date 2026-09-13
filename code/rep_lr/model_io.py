"""Rebuild a trained model from its saved ``args.json`` and load its checkpoint.

Shared by ``eval.py`` and ``inference.py`` so both reconstruct exactly the same
architecture for a given run.
"""
import torch

from models import (ComplexSequenceProjector, Encoder,
                   RFClassificationHead, ChannelEstimationHead, CFOEstimationHead,
                   SimpleCFOEstimationHead, DirectCFOEstimationHead, CFOAdaptiveHead, TaskAdaptiveEncoder)

NUM_CLASSES = 16


def build_model(train_args, num_classes=NUM_CLASSES):
    """Re-create the (untrained) architecture described by ``train_args``."""
    encoder_num_blocks = getattr(train_args, 'encoder_num_blocks', 1)

    if getattr(train_args, 'mtl', False):
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
        encoder_input_dim = train_args.proj_seq_len
        input_channels = 2
        if getattr(train_args, 'fusion_type', 'sum') == 'concat':
            encoder_input_dim = train_args.proj_seq_len * len(train_args.task)
        elif getattr(train_args, 'fusion_type', 'sum') == 'depth_concat':
            input_channels = 2 * len(train_args.task)

        if getattr(train_args, 'task_adaptive_encoder', False):
            print("Using task-adaptive encoder.")
            encoder = TaskAdaptiveEncoder(slice_size=encoder_input_dim, output_dim=train_args.d2, dropout=train_args.dropout, num_blocks=encoder_num_blocks, input_channels=input_channels)
        else:
            encoder = Encoder(slice_size=encoder_input_dim, output_dim=train_args.d2, dropout=train_args.dropout, num_blocks=encoder_num_blocks, input_channels=input_channels)

        return torch.nn.ModuleDict({
            'projections': projections,
            'encoder': encoder,
            'heads': heads
        })

    print("Reconstructing single-task model architecture.")
    task_name = train_args.task

    # Create projection layer based on task
    seq_len = train_args.slice_len if task_name == 'rf_fingerprinting' else 160
    projection = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=train_args.proj_seq_len, hidden_dim=train_args.proj_hidden_dim)

    # Choose encoder based on training arguments (Single-task)
    if getattr(train_args, 'task_adaptive_encoder', False):
        encoder = TaskAdaptiveEncoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout, num_blocks=encoder_num_blocks, input_channels=2)
    else:
        encoder = Encoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout, num_blocks=encoder_num_blocks, input_channels=2)

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

    return torch.nn.ModuleDict({
        'projection': projection,
        'encoder': encoder,
        'head': task_head
    })


def load_weights(model, ckpt_path, device):
    """Load a checkpoint saved by ``train.py`` into a model from ``build_model``."""
    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'projections' in model:
        model['projections'].load_state_dict(checkpoint['projections_state_dict'])
        model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
        model['heads'].load_state_dict(checkpoint['heads_state_dict'])
    # Single-task models saved as a single state dict
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    # Format for models where each module is saved separately
    elif all(k in checkpoint for k in ('projection_state_dict', 'encoder_state_dict', 'head_state_dict')):
        model['projection'].load_state_dict(checkpoint['projection_state_dict'])
        model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
        model['head'].load_state_dict(checkpoint['head_state_dict'])
    else:
        raise KeyError("Could not find model weights in a recognized format in the checkpoint.")
    return model
