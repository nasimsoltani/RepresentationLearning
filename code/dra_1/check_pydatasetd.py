import os
import pickle
from torch.utils.data import DataLoader
from py_datasets import ActivationDataset
from dotenv import load_dotenv
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--leaked_fraction', type=float, default=0.1)
parser.add_argument('--activation_dir', type=str, default="/scratch/10608/aadharsh_aadhithya/data/rep_lr/activations_exps/results_20250730_220537/rf")
args = parser.parse_args()


activation_dir = args.activation_dir




pkl_file_path = os.environ.get('PKL_FILE_PATH')
if not pkl_file_path:
    raise ValueError("PKL_FILE_PATH environment variable not set.")

partition_file = os.path.join(pkl_file_path, 'rf_partition_dict_0.5.pkl')
with open(partition_file, 'rb') as f:
    partitions = pickle.load(f)

train_files = partitions['train']
val_files = partitions['val']
test_files = partitions['test']

# Sub-sample the training data based on the leaked fraction
num_leaked_samples = int(len(train_files) * args.leaked_fraction)
leaked_train_files = train_files[:num_leaked_samples]
print(f"Using {len(leaked_train_files)} ({args.leaked_fraction*100}%) training samples for the attack.")


train_dataset = ActivationDataset(activation_dir=activation_dir, file_list=leaked_train_files)
val_dataset = ActivationDataset(activation_dir=activation_dir, file_list=val_files, test_mode=True)
test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)



rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, original_filename  = next(iter(train_loader))

print(activation.shape)
print(rf_x.shape)
print(rf_y.shape)
print(cfo_x.shape)
print(original_filename)
print(cfo_x)


