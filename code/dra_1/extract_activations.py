
import os
import sys
import json
import argparse
import torch
import numpy as np
import pickle
import random
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

# Adjust sys.path to allow imports from the 'rep_lr' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rep_lr.models import (ComplexSequenceProjector, Encoder, 
                   RFClassificationHead, ChannelEstimationHead, CFOEstimationHead)
from rep_lr.py_datasets import TrainDataset
from torch.utils.data import DataLoader

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
        
        # Individual batch norm layers
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

        # Calculate the size after all pooling operations (5 pooling layers)
        conv_output_size = channel * (slice_size // 32)

        # Final MLP layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(conv_output_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 2 * output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, slice_size)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2, output_dim)
        """
        # x shape: (B, 2, L)
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

def extract_activations(cli_args):
    """
    Extracts activations from the shared encoder of a trained model.
    """
    # Determine model directory from the checkpoint path
    if os.path.isdir(cli_args.model_path):
        model_dir = cli_args.model_path
        # Try to find a .pt file in the directory
        pt_files = [f for f in os.listdir(model_dir) if f.endswith('_best.pt')]
        if not pt_files:
            raise FileNotFoundError(f"No '_best.pt' file found in directory: {model_dir}")
        if len(pt_files) > 1:
            print(f"Warning: Multiple '_best.pt' files found. Using the first one: {pt_files[0]}")
        model_checkpoint_path = os.path.join(model_dir, pt_files[0])
    else:
        model_checkpoint_path = cli_args.model_path
        model_dir = os.path.dirname(model_checkpoint_path)

    # Load training arguments from the saved args.json
    args_path = os.path.join(model_dir, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {model_dir}. Cannot determine model architecture.")
    with open(args_path, 'r') as f:
        train_args = argparse.Namespace(**json.load(f))
    
    # Override with CLI args
    train_args.gpu_id = cli_args.gpu_id
    train_args.data_fraction = cli_args.data_fraction

    # Create output directory
    output_dir = cli_args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving activations to: {output_dir}")

    # Set device
    device = torch.device(f'cuda:{train_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load train data
    pkl_file_path_env = os.environ.get('PKL_FILE_PATH')
    if not pkl_file_path_env:
        raise ValueError("Environment variable PKL_FILE_PATH is not set. Please set it to the directory containing the dataset.")
    data_path = os.path.join(pkl_file_path_env, 'rf_partition_dict_0.5.pkl')
    
    print(f"Using dataset: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'")

    with open(data_path, 'rb') as handle:
        content = pickle.load(handle)
    
    # Combine all datasets: train, validation, and test
    train_data = content['train']
    val_data = content['val'] 
    test_data = content['test']
    
    data_list = train_data + val_data + test_data
    max_cfo = content.get('max_cfo')
    mean_cfo = content.get('mean_cfo')
    std_cfo = content.get('std_cfo')
    
    print(f"Dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print(f"Total samples to process: {len(data_list)}")
    
    # Subset the combined dataset if data_fraction is specified
    if train_args.data_fraction < 1.0:
        random.shuffle(data_list)
        num_samples = int(len(data_list) * train_args.data_fraction)
        data_list = data_list[:num_samples]
        print(f"Using {num_samples} samples ({train_args.data_fraction*100:.2f}% of total dataset).")
    else:
        print(f"Using all {len(data_list)} samples from train+val+test sets.")

    
    # Create ID class dict
    ID_class_dict = {f'Radio{i}': i for i in range(16)}
    num_classes = len(ID_class_dict)

    dataset = TrainDataset(data_list, ID_class_dict, train_args, max_cfo, mean_cfo, std_cfo, test_mode=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Re-create model architecture
    is_mtl = getattr(train_args, 'mtl', False)

    if is_mtl:
        print("Reconstructing MTL model architecture.")
        projections = torch.nn.ModuleDict()
        heads = torch.nn.ModuleDict() # Not used for extraction but needed for loading checkpoint
        
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
        
        model_projections = projections
        model_encoder = encoder
        model_heads = heads
        
    else: # Single-task
        print("Reconstructing single-task model architecture.")
        task_name = train_args.task
        seq_len = train_args.slice_len if task_name == 'rf_fingerprinting' else 160
        projection = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=train_args.proj_seq_len, hidden_dim=train_args.proj_hidden_dim)
        encoder = LegacyEncoder(slice_size=train_args.proj_seq_len, output_dim=train_args.d2, dropout=train_args.dropout)
        # Dummy head
        task_head = torch.nn.Identity()
        model = torch.nn.ModuleList([projection, encoder, task_head])

    # Load trained weights
    print(f"Loading weights from {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    
    if is_mtl:
        model_projections.load_state_dict(checkpoint['projections_state_dict'])
        model_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        # Heads are not needed for activation extraction, so we don't strictly need to load them
        model_projections.to(device).eval()
        model_encoder.to(device).eval()
    else:
        model[0].load_state_dict(checkpoint['module_0'])
        model[1].load_state_dict(checkpoint['module_1'])
        model.to(device).eval()

    print("Model loaded successfully. Starting activation extraction.")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Activations"):
            rf_inputs, rf_labels, cfo_inputs, cfo_labels, channel_inputs, channel_labels, file_paths = batch
            
            filename = os.path.basename(file_paths[0])
            save_path = os.path.join(output_dir, filename.replace('.mat', '.pth'))

            if os.path.exists(save_path) and not cli_args.overwrite:
                continue

            # import pdb; pdb.set_trace()

            if is_mtl:
                projected_tensors = []
                task_data_map = {
                    'rf_fingerprinting': rf_inputs,
                    'cfo_estimation': cfo_inputs,
                    'channel_estimation': channel_inputs
                }

                for task_name in train_args.task:
                    inputs = task_data_map[task_name].to(device).float()
                    if task_name == 'rf_fingerprinting':
                        # MTL eval logic for RF averages projections across slices
                        
                        proj = model_projections[task_name](inputs)
                        projected_tensors.append(proj.mean(dim=0, keepdim=True))
                    else:
                        projected_tensors.append(model_projections[task_name](inputs))
                
                projected_sum = torch.sum(torch.stack(projected_tensors), dim=0)
                encoded_activation = model_encoder(projected_sum)
                

            else: # Single-task
                task_name = train_args.task
                if task_name == 'rf_fingerprinting':
                    inputs = rf_inputs.squeeze(0).to(device).float() # Process slices
                elif task_name == 'cfo_estimation':
                    inputs = cfo_inputs.to(device).float()
                elif task_name == 'channel_estimation':
                    inputs = channel_inputs.to(device).float()

                projection, encoder, _ = model
                projected = projection(inputs)
                encoded_activation = encoder(projected)
                if task_name == 'rf_fingerprinting':
                    # Average the activations of all slices to get a single vector per file
                    encoded_activation = encoded_activation.mean(dim=0, keepdim=True)

            # Save the activation
            data_to_save = {
                'activation': encoded_activation.detach().cpu(),
                'filename': filename,
                'RF_X': rf_inputs.detach().cpu(),
                'CFO_X': cfo_inputs.detach().cpu(),
                'Channel_X': channel_inputs.detach().cpu(),
                'rf_label': rf_labels.detach().cpu(),
                'cfo_label': cfo_labels.detach().cpu(),
                'channel_label': channel_labels.detach().cpu()
            }
            torch.save(data_to_save, save_path)

    print(f"\nExtraction complete. Activations are saved in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract activations from the shared encoder of a model.')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint (.pt file) or its containing directory.')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save the extracted activation files.')
    parser.add_argument('--gpu_id', default=0, type=int, 
                        help='ID of GPU to be used.')
    parser.add_argument('--data_fraction', type=float, default=1.0, 
                        help='Fraction of the combined train+val+test dataset to use for extraction.')
    parser.add_argument('--overwrite', action='store_true',
                        help='If set, overwrite existing activation files.')

    cli_args = parser.parse_args()
    extract_activations(cli_args) 