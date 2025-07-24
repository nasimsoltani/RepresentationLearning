
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

from rep_lr.py_datasets import TrainDataset
from torch.utils.data import DataLoader
from dra_1.model_loader import load_model_for_extraction

def extract_activations(cli_args):
    """
    Extracts activations from the shared encoder of a trained model.
    """
    # Determine experiment directory from the model path
    if os.path.isdir(cli_args.model_path):
        experiment_dir = cli_args.model_path
    else:
        experiment_dir = os.path.dirname(cli_args.model_path)

    # Create output directory
    output_dir = cli_args.output_dir
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, 'activations')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving activations to: {output_dir}")

    # Set device
    device = torch.device(f'cuda:{cli_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model using the new utility
    model_data = load_model_for_extraction(experiment_dir, device)
    model = model_data['model']
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']
    
    # Override with CLI args
    train_args.gpu_id = cli_args.gpu_id
    train_args.data_fraction = cli_args.data_fraction

    print(f"Model loaded successfully. MTL: {is_mtl}")

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

    print("Starting activation extraction.")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Activations"):
            rf_inputs, rf_labels, cfo_inputs, cfo_labels, channel_inputs, channel_labels, file_paths = batch
            
            filename = os.path.basename(file_paths[0])
            save_path = os.path.join(output_dir, filename.replace('.mat', '.pth'))

            if os.path.exists(save_path) and not cli_args.overwrite:
                continue

            # This tensor will hold the RF data that gets saved.
            rf_inputs_to_save = rf_inputs.detach().cpu() #Bx2x1024
            #print(rf_inputs_to_save.shape)

            if is_mtl:
                # MTL model: use projections dict and sum projections
                projected_tensors = []
                task_data_map = {
                    'rf_fingerprinting': rf_inputs,
                    'cfo_estimation': cfo_inputs,
                    'channel_estimation': channel_inputs
                }

                for task_name in train_args.task:
                    inputs = task_data_map[task_name].to(device).float()
                    if task_name == 'rf_fingerprinting':
                       # print(f"RF inputs shape: {inputs.shape}")
                        # MTL eval logic for RF averages projections across slices
                        proj = model['projections'][task_name](inputs)
                        projected_tensors.append(proj)
                    else:
                        proj = model['projections'][task_name](inputs)
                        projected_tensors.append(proj)
                #import pdb; pdb.set_trace()
                projected_sum = torch.sum(torch.stack(projected_tensors), dim=0)
                encoded_activation = model['encoder'](projected_sum)

            else:
                # Single-task model: use projection and encoder directly
                task_name = train_args.task
                if task_name == 'rf_fingerprinting':
                    inputs = rf_inputs.to(device).float()  
                elif task_name == 'cfo_estimation':
                    inputs = cfo_inputs.to(device).float()
                elif task_name == 'channel_estimation':
                    inputs = channel_inputs.to(device).float()

                projected = model['projection'](inputs)
                
                # Handle direct CFO case where encoder might be bypassed
                if task_name == 'cfo_estimation' and getattr(train_args, 'direct_cfo', False):
                    # For direct CFO, the "encoded" activation is actually just the projection
                    # But we still want to save something, so let's use a dummy encoder pass
                    # or just use the projection directly
                    encoded_activation = projected  # Skip encoder for direct CFO
                else:
                    encoded_activation = model['encoder'](projected)
                
                # if task_name == 'rf_fingerprinting':
                #     # Average the activations of all slices to get a single vector per file
                #     encoded_activation = encoded_activation.mean(dim=0, keepdim=True)

            # Save the activation
            data_to_save = {
                'activation': encoded_activation.detach().cpu(),
                'filename': filename,
                'RF_X': rf_inputs_to_save,
                'CFO_X': cfo_inputs.detach().cpu(),
                'Channel_X': channel_inputs.detach().cpu(),
                'rf_label': rf_labels.detach().cpu(),
                'cfo_label': cfo_labels.detach().cpu(),
                'channel_label': channel_labels.detach().cpu()
            }
            torch.save(data_to_save, save_path, _use_new_zipfile_serialization=False)

    print(f"\nExtraction complete. Activations are saved in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract activations from the shared encoder of a model.')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint (.pt file) or its containing directory.')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save the extracted activation files. Defaults to <model_path>/activations.')
    parser.add_argument('--gpu_id', default=0, type=int, 
                        help='ID of GPU to be used.')
    parser.add_argument('--data_fraction', type=float, default=1.0, 
                        help='Fraction of the combined train+val+test dataset to use for extraction.')
    parser.add_argument('--overwrite', action='store_true',
                        help='If set, overwrite existing activation files.')

    cli_args = parser.parse_args()
    extract_activations(cli_args) 