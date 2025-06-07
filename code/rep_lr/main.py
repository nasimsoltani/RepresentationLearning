# import the required packages
import os
import pickle
from tqdm import tqdm
import numpy as np
import glob
import random
import argparse

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader, random_split
from py_datasets import TrainDataset
from models import RFFingerprintingNet, ChannelNet, CFONetLarge, CFONetSmall

import warnings
import wandb
from datetime import datetime

from train import train_model

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Train and validation pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Task and data arguments
    parser.add_argument('--task', type=str, required=True, 
                        choices=['rf_fingerprinting', 'channel_estimation', 'cfo_estimation_small', 'cfo_estimation_large'], 
                        help='Task to train.')
    parser.add_argument('--pkl_dataset_path', type=str, required=True, 
                        help='Path to the pkl dataset file.')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--slice_len', type=int, default=1024, help='NN input size for RF fingerprinting NN.')
    parser.add_argument('--num_classes', type=int, default=16, help='Number of classes for RF fingerprinting.')

    # System and logging
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used.')
    parser.add_argument('--save_path', type=str, default='results', help='Base directory to save results.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--save_epochs', type=int, default=10, help='Save checkpoint every N epochs. 0 to disable.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping.')
    parser.add_argument('--wandb_project', type=str, default='representation-learning-lab', help='WandB project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity.')

    args = parser.parse_args()

    # Create a unique directory for this run
    run_name = f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_path = os.path.join(args.save_path, run_name)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Results will be saved to: {args.save_path}")

    # Initial configurations
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, name=run_name)

    # Create ID class dict 
    ID_class_dict = {}
    for i in range(16):
        this_key = 'Radio'+str(i)
        ID_class_dict[this_key] = i
    print(ID_class_dict)

    # Load data from pickle file
    with open(args.pkl_dataset_path, 'rb') as handle:
        content = pickle.load(handle)

    train_list = content['train']
    val_list = content['val']

    dataset_args = argparse.Namespace(slice_len=args.slice_len)
    train_dataset = TrainDataset(train_list, ID_class_dict, dataset_args)
    val_dataset = TrainDataset(val_list, ID_class_dict, dataset_args)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model and loss function selection
    if args.task == 'rf_fingerprinting':
        model = RFFingerprintingNet(slice_size=args.slice_len, num_classes=args.num_classes)
        loss_fn = nn.CrossEntropyLoss()
    elif args.task == 'channel_estimation':
        model = ChannelNet()
        loss_fn = nn.MSELoss()
    elif args.task == 'cfo_estimation_small':
        model = CFONetSmall()
        loss_fn = nn.MSELoss()
    elif args.task == 'cfo_estimation_large':
        model = CFONetLarge()
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {args.task}")

    model.to(device)
    
    # Load from checkpoint if provided
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint '{args.resume_from}'")
            model.load_state_dict(torch.load(args.resume_from, map_location=device))
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
# for RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y in train_dl:
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



 

