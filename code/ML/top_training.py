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

from dataset_partitioner import dataset_partitioner
from data_generator import TrainDataset, SpectrogramTrainDataset
from model import IQNet, SpectrogramNet #, TransformerModel

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = 'Train and validation pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used')
parser.add_argument('--ood_classes', default='', type=str, help='OOD classes separated with comma')
parser.add_argument('--augmentation', action='store_true', help='whether to do augmentation')
parser.add_argument('--fft', action='store_true', help='whether to feed in fft or time domain samples')
parser.add_argument('--spectrogram', action='store_true', help='whether to feed in fft or time domain samples')


"""parser.add_argument('--aug_type', default='noaug', type=str, help='options: noaug, flipaug, rowshuffleaug, sectionshuffleaug')
parser.add_argument('--Round', default='Round', type=str, help='options: Round0, Round1, Round2')
parser.add_argument('--buffer_reduction', default='0.0', type=str, help='options: Round1, Round2')
parser.add_argument('--dataset_base', default='/home/nasim/', type=str, help='include the path to your ORAN code folder')
parser.add_argument('--dependent_on_previous_round', action='store_true', help='is this round going to use data from previous round and rehearsal buffer?')
"""
 
args = parser.parse_args()

save_path = '/home/ns38942/AiR/results/'
# path to the folder that contains all dataset files :
dataset_folder_path = '/home/ns38942/AiR/dataset/baseline-interferers-with-channels-v1/'
pkl_dataset_path = '/home/ns38942/AiR/pkls/'
slice_len = 2048 #4096*2	   #2048    #512 
batch_size = 256 
epochs = 300           # Number of epochs you want to train for
early_stopping = True  # Set to True or False to enable or disable early stopping
# If early_stopping is enabled, patience shows the number of consecutive epochs
# after which training stops if training loss does not improve.
patience = 20 
#class_ids = {'BPSK':0, 'CW':1, 'FAWGN':2, 'FHCW':3, 'LFM':4, 'LFM2':5, 'OOK':6, 'PCW':7}  #1600 samples in each class
#OOD_classes = ['LFM','LFM2','PCW']

if args.spectrogram:
	batch_size  = 32

class_ids = {'AM':0, 'BOCRBPSK':1, 'CW':2, 'FAWGN':3, 'FHCW':4, 'LFM':5, 'PCW':6, 'RBPSK':7, 'ROOK':8}  #1600 samples in each class
OOD_classes = args.ood_classes.split(',')


ID_classes = list(filter(lambda x: x not in OOD_classes, list(class_ids.keys())))
print(ID_classes)


# create filename_suffix for saving weights and loss
filename_suffix = 'OOD'
for OOD_class in OOD_classes:
	filename_suffix += '_'+OOD_class 

dataset_path = os.path.join(pkl_dataset_path , 'dataset_' + filename_suffix + '.pkl') 
print(filename_suffix)

# Initial configurations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# create ID class dict 
ID_class_dict = {}
dummy_counter = 0
for class_name in class_ids:
	if class_name in ID_classes:
		ID_class_dict[class_name] = dummy_counter
		dummy_counter += 1
print(ID_class_dict)

# partition
#train_list, val_list, labels = dataset_partitioner(dataset_folder_path, dataset_path, class_ids, OOD_classes)


#### Read the dataset pkl file here ########


if args.spectrogram:
	train_dataset = SpectrogramTrainDataset(train_list, labels, ID_class_dict, args, slice_len, {}, 1)
	val_dataset = SpectrogramTrainDataset(val_list, labels, ID_class_dict, args, slice_len, {}, 1)
else:
	train_dataset = TrainDataset(train_list, labels, ID_class_dict, args, slice_len, {}, 1)
	val_dataset = TrainDataset(val_list, labels, ID_class_dict, args, slice_len, {}, 1)

# test_dataset = TestDataset(test_list, labels, class_ids)
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
# test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

print('length of train and val dl')
print(len(train_dl))
print(len(val_dl))
for anchor, positive, negative, y in train_dl:
	print(anchor.shape, positive.shape, negative.shape)
	num_channels = anchor.shape[1]
	#print(y)
	break

num_classes = len(list(ID_class_dict.keys()))

if args.spectrogram:
	model = SpectrogramNet(slice_len, num_channels, num_classes)
else:
	model = IQNet(slice_len, num_channels, num_classes)
#elif args.model == 'Transformer':
#model = TransformerModel(classes=num_classes, d_model=64*2, seq_len=seq_len, nlayers=2, use_pos=False)


# print number of parameters in the model
pp=0
for p in list(model.parameters()):
    n=1
    for s in list(p.size()):
        n = n*s
    pp += n
print('This model has ' +str(pp)+ ' parameters')
    
train_model(model, train_dl, val_dl, epochs, save_path, early_stopping, patience, filename_suffix)
