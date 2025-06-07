import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from py_datasets import TrainDataset
from mtl_models import EarlyFusionNet
from models import RFFingerprintingNet, ChannelNet, CFONetLarge, CFONetSmall
from mtl_train import train_mtl_model
import wandb
from datetime import datetime
from itertools import chain

def main():
    parser = argparse.ArgumentParser(description='Multi-task learning training pipeline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data arguments
    parser.add_argument('--pkl_dataset_path', type=str, required=True, help='Path to the pkl dataset file.')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--slice_len', type=int, default=1024, help='NN input size for RF fingerprinting NN.')
    parser.add_argument('--num_classes', type=int, default=16, help='Number of classes for RF fingerprinting.')
    parser.add_argument('--common_rep_dim', type=int, default=512, help='Dimension of common representation.')
    parser.add_argument('--cfo_net_type', type=str, default='large', choices=['small', 'large'], help='Type of CFO network to use.')

    # Loss weights
    parser.add_argument('--loss_weight_rf', type=float, default=1.0, help='Weight for RF fingerprinting loss.')
    parser.add_argument('--loss_weight_channel', type=float, default=1.0, help='Weight for channel estimation loss.')
    parser.add_argument('--loss_weight_cfo', type=float, default=1.0, help='Weight for CFO estimation loss.')

    # System and logging
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used.')
    parser.add_argument('--save_path', type=str, default='results_mtl', help='Base directory to save results.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--save_epochs', type=int, default=10, help='Save checkpoint every N epochs. 0 to disable.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping.')
    parser.add_argument('--wandb_project', type=str, default='representation-learning-lab-mtl', help='WandB project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity.')

    args = parser.parse_args()

    # Create a unique directory for this run
    run_name = f"mtl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_path = os.path.join(args.save_path, run_name)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Results will be saved to: {args.save_path}")
    
    # Initial configurations
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, name=run_name)
    
    # Load data
    with open(args.pkl_dataset_path, 'rb') as handle:
        content = pickle.load(handle)
    train_list = content['train']
    val_list = content['val']

    ID_class_dict = {f'Radio{i}': i for i in range(16)}
    dataset_args = argparse.Namespace(slice_len=args.slice_len)
    train_dataset = TrainDataset(train_list, ID_class_dict, dataset_args)
    val_dataset = TrainDataset(val_list, ID_class_dict, dataset_args)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Models
    fusion_model = EarlyFusionNet(common_rep_dim=args.common_rep_dim)
    rf_model = RFFingerprintingNet(slice_size=args.slice_len, num_classes=args.num_classes, mtl=True, common_rep_dim=args.common_rep_dim)
    channel_model = ChannelNet(mtl=True, common_rep_dim=args.common_rep_dim)
    if args.cfo_net_type == 'large':
        cfo_model = CFONetLarge(mtl=True, common_rep_dim=args.common_rep_dim)
    else:
        cfo_model = CFONetSmall(mtl=True, common_rep_dim=args.common_rep_dim)

    models = {'fusion': fusion_model, 'rf': rf_model, 'channel': channel_model, 'cfo': cfo_model}

    # Loss functions
    loss_fns = {
        'rf': nn.CrossEntropyLoss(),
        'channel': nn.MSELoss(),
        'cfo': nn.MSELoss()
    }
    
    # Optimizer
    optimizer = optim.Adam(
        chain(fusion_model.parameters(), rf_model.parameters(), channel_model.parameters(), cfo_model.parameters()),
        lr=args.lr
    )
    
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint '{args.resume_from}'")
            checkpoint = torch.load(args.resume_from, map_location=device)
            fusion_model.load_state_dict(checkpoint['fusion_model'])
            rf_model.load_state_dict(checkpoint['rf_model'])
            channel_model.load_state_dict(checkpoint['channel_model'])
            cfo_model.load_state_dict(checkpoint['cfo_model'])
        else:
            print(f"Checkpoint not found at '{args.resume_from}'. Training from scratch.")

    # Start training
    train_mtl_model(models, train_dl, val_dl, loss_fns, optimizer, args)
    
    wandb.finish()

if __name__ == '__main__':
    main() 