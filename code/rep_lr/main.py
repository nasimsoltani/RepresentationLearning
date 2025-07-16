# import the required packages
import os
import pickle
from tqdm import tqdm
import numpy as np
import glob
import random
import argparse
import json

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader, random_split
from py_datasets import TrainDataset
from models import (ComplexSequenceProjector, Encoder, 
                   RFClassificationHead, ChannelEstimationHead, CFOEstimationHead)

import warnings
import wandb
from datetime import datetime

from train import train_model

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Train and validation pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Task and data arguments
    parser.add_argument('--task', type=str, required=True, nargs='+',
                        choices=['rf_fingerprinting', 'channel_estimation', 'cfo_estimation'],
                        help='Task(s) to train. For MTL, provide multiple tasks.')
    parser.add_argument('--pkl_dataset_path', type=str, required=True, 
                        help='Path to the pkl dataset file.')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--slice_len', type=int, default=1024, help='NN input size. 1024 for RF, 160 for CFO/Channel.')

    # Architecture parameters
    parser.add_argument('--proj_seq_len', type=int, default=256,
                        help='Common sequence length after projection.')
    parser.add_argument('--proj_hidden_dim', type=int, default=512,
                        help='Hidden dimension for projection layer MLP.')
    parser.add_argument('--d2', type=int, default=128, 
                        help='Output dimension of encoder.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability for all layers.')
    parser.add_argument('--head_hidden_dim', type=int, default=256,
                        help='Hidden dimension for task head.')
    
    # MTL arguments
    parser.add_argument('--mtl', action='store_true', help='Enable Multi-Task Learning.')
    parser.add_argument('--w_rf', type=float, default=1.0, help='Weight for RF fingerprinting loss.')
    parser.add_argument('--w_channel', type=float, default=1.0, help='Weight for channel estimation loss.')
    parser.add_argument('--w_cfo', type=float, default=1.0, help='Weight for CFO estimation loss.')

    # System and logging
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used.')
    parser.add_argument('--save_path', type=str, default='results', help='Base directory to save results.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--save_epochs', type=int, default=10, help='Save checkpoint every N epochs. 0 to disable.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping.')
    parser.add_argument('--wandb_project', type=str, default='representation-learning-lab', help='WandB project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity.')

    args = parser.parse_args()

    if not args.mtl and len(args.task) > 1:
        raise ValueError("Multiple tasks specified without --mtl flag. Use --mtl for multi-task learning.")
    if not args.mtl:
        # For backward compatibility and simplicity in single-task mode
        args.task = args.task[0]

    # Create a unique directory for this run
    run_name = f"{'_'.join(args.task) if args.mtl else args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_path = os.path.join(args.save_path, run_name)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Results will be saved to: {args.save_path}")

    # Save args to a JSON file
    args_dict = vars(args)
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    # Initial configurations
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb
    wandb.init(project=args.wandb_project, config=args, name=run_name)

    # Create ID class dict 
    ID_class_dict = {}
    for i in range(16):
        this_key = 'Radio'+str(i)
        ID_class_dict[this_key] = i
    print(ID_class_dict)
    num_classes = len(list(ID_class_dict.keys()))

    # Load data from pickle file
    with open(args.pkl_dataset_path, 'rb') as handle:
        content = pickle.load(handle)

    train_list = content['train']
    val_list = content['val']
    max_cfo = content['max_cfo']
    mean_cfo = content['mean_cfo']
    std_cfo = content['std_cfo']


    
    dataset_args = argparse.Namespace(slice_len=args.slice_len)
    train_dataset = TrainDataset(train_list, ID_class_dict, dataset_args, max_cfo, mean_cfo, std_cfo)
    val_dataset = TrainDataset(val_list, ID_class_dict, dataset_args, max_cfo, mean_cfo, std_cfo)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create model, loss function, and optimizer
    model = None
    loss_fn = None

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    if args.mtl:
        # Multi-Task Learning Setup
        print("Setting up Multi-Task Learning model for tasks:", args.task)
        
        projections = nn.ModuleDict()
        heads = nn.ModuleDict()
        loss_fns = {}
        
        for task in args.task:
            if task == 'rf_fingerprinting':
                seq_len = args.slice_len
                projections[task] = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=args.proj_seq_len, hidden_dim=args.proj_hidden_dim)
                heads[task] = RFClassificationHead(input_dim=2*args.d2, num_classes=num_classes, hidden_dim=args.head_hidden_dim, dropout=args.dropout)
                loss_fns[task] = nn.CrossEntropyLoss()

            elif task == 'channel_estimation':
                seq_len = 160
                projections[task] = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=args.proj_seq_len, hidden_dim=args.proj_hidden_dim)
                heads[task] = ChannelEstimationHead(input_dim=2*args.d2, hidden_dim=args.head_hidden_dim, output_length=52, dropout=args.dropout)
                mse_loss = nn.MSELoss()
                def complex_mse_loss(pred, target):
                    pred_flat = pred.view(pred.size(0), -1)
                    target_flat = target.view(target.size(0), -1)
                    return mse_loss(pred_flat, target_flat)
                loss_fns[task] = complex_mse_loss

            elif task == 'cfo_estimation':
                seq_len = 160
                projections[task] = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=args.proj_seq_len, hidden_dim=args.proj_hidden_dim)
                heads[task] = CFOEstimationHead(input_dim=2*args.d2, hidden_dim=args.head_hidden_dim, dropout=args.dropout)
                loss_fns[task] = nn.MSELoss()

        encoder = Encoder(slice_size=args.proj_seq_len, output_dim=args.d2, dropout=args.dropout)
        
        model = nn.ModuleDict({
            'projections': projections,
            'encoder': encoder,
            'heads': heads
        })
        loss_fn = loss_fns

        # Print model parameters
        print("\nModel Architecture Details (MTL):")
        for task_name, proj in projections.items():
            print(f"Projection Parameters ({task_name}): {count_parameters(proj):,}")
        print(f"Encoder Parameters: {count_parameters(encoder):,}")
        for task_name, head in heads.items():
            print(f"Task Head Parameters ({task_name}): {count_parameters(head):,}")
        print(f"Total Parameters: {count_parameters(model):,}")

    else:
        # Single-Task Learning Setup
        print(f"Setting up Single-Task Learning model for {args.task}.")
        # Create projection layer
        seq_len = args.slice_len if args.task == 'rf_fingerprinting' else 160
        
        projection = ComplexSequenceProjector(input_seq_len=seq_len, output_seq_len=args.proj_seq_len, hidden_dim=args.proj_hidden_dim)
        
        # Create encoder (common for all tasks)
        encoder = Encoder(slice_size=args.proj_seq_len, output_dim=args.d2, dropout=args.dropout)

        # Task-specific head and loss function
        if args.task == 'rf_fingerprinting':
            task_head = RFClassificationHead(
                input_dim=2*args.d2, 
                num_classes=num_classes, 
                hidden_dim=args.head_hidden_dim, 
                dropout=args.dropout
            )
            loss_fn = nn.CrossEntropyLoss()
            
        elif args.task == 'channel_estimation':
            task_head = ChannelEstimationHead(
                input_dim=2*args.d2,
                hidden_dim=args.head_hidden_dim,
                output_length=52,
                dropout=args.dropout
            )
            mse_loss = nn.MSELoss()
            def complex_mse_loss(pred, target):
                pred_flat = pred.view(pred.size(0), -1)
                target_flat = target.view(target.size(0), -1)
                return mse_loss(pred_flat, target_flat)
            loss_fn = complex_mse_loss
            
        elif args.task == 'cfo_estimation':
            task_head = CFOEstimationHead(
                input_dim=2*args.d2,
                hidden_dim=args.head_hidden_dim,
                dropout=args.dropout
            )
            loss_fn = nn.MSELoss()
            
        else:
            raise ValueError(f"Unknown task: {args.task}")

        model = nn.ModuleList([projection, encoder, task_head])
        
        total_params = count_parameters(projection) + count_parameters(encoder) + count_parameters(task_head)
        print("\nModel Architecture Details:")
        print(f"Projection Parameters: {count_parameters(projection):,}")
        print(f"Encoder Parameters: {count_parameters(encoder):,}")
        print(f"Task Head Parameters: {count_parameters(task_head):,}")
        print(f"Total Parameters: {total_params:,}")

    # The rest of the script remains largely the same, but checkpointing needs to be handled.
    # We can remove the old complex memory calculation for now to simplify.
    print("\nNote: Memory usage calculation has been simplified.\n")

    # Move model to device
    model.to(device)
    
    # Load from checkpoint if provided
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint '{args.resume_from}'")
            checkpoint = torch.load(args.resume_from, map_location=device)
            if args.mtl:
                model['projections'].load_state_dict(checkpoint['projections_state_dict'])
                model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
                model['heads'].load_state_dict(checkpoint['heads_state_dict'])
            else:
                if 'model_state_dict' in checkpoint: # Handle old single-task checkpoints
                    model.load_state_dict(checkpoint['model_state_dict'])
                else: # Handle new single-task checkpoints (saved as ModuleList components)
                    for i, module in enumerate(model):
                        module.load_state_dict(checkpoint[f'module_{i}'])
            print("Checkpoint loaded.")
        else:
            print(f"Checkpoint not found at '{args.resume_from}'. Training from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Start training
    train_model(model, train_dl, val_dl, loss_fn, optimizer, args)

    wandb.finish()

if __name__ == '__main__':
    main()

# print('length of train and val dl')
# print(len(train_dl))
# print(len(val_dl))
# for RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y, _ in train_dl:
# 	print("Shapes from Dataloader batch:")
# 	print(f"	RF_X shape: {RF_X.shape}")
# 	print(f"	RF_y shape: {RF_y.shape}")
# 	print(f"	CFO_X shape: {CFO_X.shape}")
# 	print(f"	CFO_y shape: {CFO_y.shape}")
# 	print(f"	Channel_X shape: {Channel_X.shape}")
# 	print(f"	Channel_y shape: {Channel_y.shape}")
# 	break

# print('\n--- Testing ChannelNet ---')
# channel_model = ChannelNet()
# # print number of parameters in the model
# pp_channel = 0
# for p in list(channel_model.parameters()):
#     n=1
#     for s in list(p.size()):
#         n = n*s
#     pp_channel += n
# print('ChannelNet model has ' +str(pp_channel)+ ' parameters')

# print("Input shape for ChannelNet:", Channel_X.shape)
# channel_output = channel_model(Channel_X)
# print("Output shape from ChannelNet:", channel_output.shape)
# print("--- ChannelNet Test End ---\n")

# print('\n--- Testing CFONetLarge ---')
# cfo_model_large = CFONetLarge()
# pp_cfo_large = 0
# for p in list(cfo_model_large.parameters()):
#     n=1
#     for s in list(p.size()):
#         n = n*s
#     pp_cfo_large += n
# print('CFONetLarge model has ' +str(pp_cfo_large)+ ' parameters')

# print("Input shape for CFONetLarge:", CFO_X.shape)
# cfo_output_large = cfo_model_large(CFO_X)
# print("Output shape from CFONetLarge:", cfo_output_large.shape)
# print("--- CFONetLarge Test End ---\n")

# print('\n--- Testing CFONetSmall ---')
# cfo_model_small = CFONetSmall()
# pp_cfo_small = 0
# for p in list(cfo_model_small.parameters()):
#     n=1
#     for s in list(p.size()):
#         n = n*s
#     pp_cfo_small += n
# print('CFONetSmall model has ' +str(pp_cfo_small)+ ' parameters')

# print("Input shape for CFONetSmall:", CFO_X.shape)
# cfo_output_small = cfo_model_small(CFO_X)
# print("Output shape from CFONetSmall:", cfo_output_small.shape)
# print("--- CFONetSmall Test End ---\n")

# num_classes = len(list(ID_class_dict.keys()))
# rf_net = RFFingerprintingNet(args.slice_len, num_classes)

# print('\n--- Testing RFFingerprintingNet ---')
# # print number of parameters in the model
# pp_channel = 0
# for p in list(rf_net.parameters()):
#     n=1
#     for s in list(p.size()):
#         n = n*s
#     pp_channel += n
# print('RFFingerprintingNet model has ' +str(pp_channel)+ ' parameters')

# print("Input shape for ChannelNet:", RF_X.shape)
# rf_output = rf_net(RF_X)
# print("Output shape from ChannelNet:", rf_output.shape)
# print("--- RFFingerprintingNet Test End ---\n")



 

