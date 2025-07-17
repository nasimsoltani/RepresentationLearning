import os
import sys
import json
import argparse
import torch
import torch.nn as nn

# Adjust sys.path to allow imports from the 'rep_lr' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rep_lr.models import (ComplexSequenceProjector, UpsamplingProjector, 
                           RFClassificationHead, ChannelEstimationHead, CFOEstimationHead)

class LegacyEncoder(torch.nn.Module):
    """
    Legacy encoder class that matches the saved model structure with individual conv layers.
    This is a compatibility layer for loading older model checkpoints.
    """
    def __init__(self, slice_size, output_dim=128, dropout=0.25):
        super(LegacyEncoder, self).__init__()
        self.output_dim = output_dim
        channel = 64

        # Individual conv layers to match saved model structure
        self.conv0 = torch.nn.Conv1d(2, channel, kernel_size=7, padding="same")
        self.conv1 = torch.nn.Conv1d(channel, channel, kernel_size=5, padding="same")
        self.conv2 = torch.nn.Conv1d(channel, channel, kernel_size=7, padding="same")
        self.conv3 = torch.nn.Conv1d(channel, channel, kernel_size=5, padding="same")
        self.conv4 = torch.nn.Conv1d(channel, channel, kernel_size=7, padding="same")
        self.conv5 = torch.nn.Conv1d(channel, channel, kernel_size=5, padding="same")
        self.conv6 = torch.nn.Conv1d(channel, channel, kernel_size=7, padding="same")
        self.conv7 = torch.nn.Conv1d(channel, channel, kernel_size=5, padding="same")
        self.conv8 = torch.nn.Conv1d(channel, channel, kernel_size=7, padding="same")
        self.conv9 = torch.nn.Conv1d(channel, channel, kernel_size=5, padding="same")
        
        self.bn0 = torch.nn.BatchNorm1d(channel)
        self.bn1 = torch.nn.BatchNorm1d(channel)
        self.bn2 = torch.nn.BatchNorm1d(channel)
        self.bn3 = torch.nn.BatchNorm1d(channel)
        self.bn4 = torch.nn.BatchNorm1d(channel)
        self.bn5 = torch.nn.BatchNorm1d(channel)
        self.bn6 = torch.nn.BatchNorm1d(channel)
        self.bn7 = torch.nn.BatchNorm1d(channel)
        self.bn8 = torch.nn.BatchNorm1d(channel)
        self.bn9 = torch.nn.BatchNorm1d(channel)

        self.pool = torch.nn.MaxPool1d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1)

        conv_output_size = channel * (slice_size // 32)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(conv_output_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 2 * output_dim)
        )

    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.pool(x)
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.pool(x)
        features = self.flatten(x)
        output = self.classifier(features)
        return output.view(output.size(0), 2, self.output_dim)

def load_trained_model(model_checkpoint_path, device):
    """
    Loads a pre-trained model (projections, encoder, heads) from a checkpoint.
    This function reconstructs the model architecture based on the saved 'args.json'.
    """
    model_dir = os.path.dirname(model_checkpoint_path)
    args_path = os.path.join(model_dir, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {model_dir}")

    with open(args_path, 'r') as f:
        train_args = argparse.Namespace(**json.load(f))

    is_mtl = getattr(train_args, 'mtl', False)
    
    # This is needed for RF head but not stored in args.json, so we set a default.
    num_classes = 16 

    if is_mtl:
        projections = nn.ModuleDict()
        heads = nn.ModuleDict()
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
                heads[task] = CFOEstimationHead(input_dim=2*train_args.d2, hidden_dim=train_args.head_hidden_dim, dropout=train_args.dropout)
        
        encoder = LegacyEncoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout)
        model = {'projections': projections, 'encoder': encoder, 'heads': heads}

    else: # Single-task
        task_name = train_args.task[0] if isinstance(train_args.task, list) else train_args.task
        seq_len = train_args.slice_len if task_name == 'rf_fingerprinting' else 160
        projection = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=train_args.proj_seq_len, hidden_dim=train_args.proj_hidden_dim)
        encoder = LegacyEncoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout)
        # The head is not used in the attack, so a dummy is fine.
        head = nn.Identity()
        model = {'projection': projection, 'encoder': encoder, 'head': head}

    # Load weights
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    if is_mtl:
        model['projections'].load_state_dict(checkpoint['projections_state_dict'])
        model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
        # heads are not needed for the attack
    else:
        # Compatibility with different single-task checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Create a full model dict to load into
            full_model_dict = nn.ModuleDict(model)
            full_model_dict.load_state_dict(checkpoint['model_state_dict'])
        elif 'module_0' in checkpoint and 'module_1' in checkpoint: # Format from extract_activations
            model['projection'].load_state_dict(checkpoint['module_0'])
            model['encoder'].load_state_dict(checkpoint['module_1'])
        else:
            raise KeyError("Unrecognized single-task checkpoint format.")

    # Freeze model and set to eval mode
    for component in model.values():
        if isinstance(component, nn.Module):
            component.to(device)
            component.eval()
            for param in component.parameters():
                param.requires_grad = False

    print("Trained model loaded and frozen successfully.")
    return model, train_args, is_mtl 