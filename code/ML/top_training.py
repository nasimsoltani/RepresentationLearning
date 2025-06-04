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
from torch.utils.data import Dataset, DataLoader
from train_model import train_model

from data_generator import TrainDataset
from model import RFFingerprintingNet

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = 'Train and validation pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used')
parser.add_argument('--slice_len', default=1024, type=int, help='NN input size for RF fingerprinting NN')
parser.add_argument('--pkl_dataset_path', default='', type=str, help='Path to pkl dataset')
parser.add_argument('--save_path', default='', type=str, help='Path to result folder where you want to save trained models')


args = parser.parse_args()

batch_size = 256 
epochs = 300           # Number of epochs you want to train for
early_stopping = True  # Set to True or False to enable or disable early stopping
# If early_stopping is enabled, patience shows the number of consecutive epochs
# after which training stops if training loss does not improve.
patience = 20 


# Initial configurations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# create ID class dict 
ID_class_dict = {}
for i in range(16):
	this_key = 'Radio'+str(i)
	ID_class_dict[this_key] = i
print(ID_class_dict)


#### Read the dataset pkl file here ########
with open (args.pkl_dataset_path, 'rb') as handle:
	content = pickle.load(handle)

train_list = content['train']
val_list = content['val']

train_dataset = TrainDataset(train_list, ID_class_dict, args)
val_dataset = TrainDataset(val_list, ID_class_dict, args)

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


print('length of train and val dl')
print(len(train_dl))
print(len(val_dl))
for RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y in train_dl:
	print(RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y)
	break

num_classes = len(list(ID_class_dict.keys()))

model = RFFingerprintingNet(args.slice_len, num_classes)

# print number of parameters in the model
pp=0
for p in list(model.parameters()):
    n=1
    for s in list(p.size()):
        n = n*s
    pp += n
print('This model has ' +str(pp)+ ' parameters')
   

""" Hi Aadharsh, this code works until here, you need to revise the train_model function that is in train_model.py, preserving the same style, but adding the 2 additional tasks"""

 
train_model(model, train_dl, val_dl, epochs, args.save_path, early_stopping, patience)
