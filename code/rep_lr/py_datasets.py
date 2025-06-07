# Because we have slicing in the IQ pipeline, we do not need a TestDataset
# We just create a TrainDataset class
# slicing during the test phase is done in the testing cell out of a data generator class

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
from scipy import signal
import torch.nn.functional as F
import torch
from torch.fft import fft as FFT
from torch.fft import fftshift as FFTshift
import os
from scipy.io import loadmat 

def read_file(file_path):
	""" gets a file_path for RF fingerprinting input part, and return associated data parts
	for CFO estimation and channel estimation too """

	# file_path is the RFfingerprinting file path, read all 3 paths
	content = loadmat(file_path)
	RF_input = torch.from_numpy(content['Packet'])[:,0]
	RF_output = content['Radio'][0]

	file_path_list = file_path.split('/')
	file_path_list.pop(0)
	filename = file_path_list[-1]
	file_path_list.pop()
	
	suffix_filename = filename.lstrip('RFfingerprinting')

	# for CFO filepath
	new_filename = '/CFOEstimation'+suffix_filename
	new_filepath = ''
	for element in file_path_list:
		new_filepath +='/'+ element
	new_filepath += new_filename
	content = loadmat(new_filepath)
	CFO_input = torch.from_numpy(content['LSTF'])[:,0]
	CFO_output = torch.from_numpy(content['CFO'])
	# for Channel Estimation filepath
	new_filename = '/ChannelEstimation'+suffix_filename
	new_filepath = ''
	for element in file_path_list:
		new_filepath +='/'+ element
	new_filepath += new_filename
	content = loadmat(new_filepath)
	Channel_input = torch.from_numpy(content['LLTF'])[:,0]
	Channel_output = torch.from_numpy(content['EstChannnel'])[:,0]
	# normalize everything and send out
	# calculate RMS and normalize the IQ sequence
	RMS = torch.sqrt(torch.mean(torch.real(RF_input)**2+torch.imag(RF_input)**2))
	RF_input = RF_input/RMS
	RMS = torch.sqrt(torch.mean(torch.real(Channel_input)**2+torch.imag(Channel_input)**2))
	Channel_input = Channel_input/RMS
	RMS = torch.sqrt(torch.mean(torch.real(Channel_output)**2+torch.imag(Channel_output)**2))
	Channel_output = Channel_output/RMS
	RMS = torch.sqrt(torch.mean(torch.real(CFO_input)**2+torch.imag(CFO_input)**2))
	CFO_input = CFO_input/RMS
	# separate I and Q as two separate channels and bring the channels
	# to the first dimension for pytorch compatibility
	temp = torch.zeros((2,RF_input.shape[0]))
	temp[0,:] = torch.real(RF_input)
	temp[1,:] = torch.imag(RF_input)
	RF_X = temp
	RF_y = int(RF_output.lstrip('Radio'))

	temp = torch.zeros((2,Channel_input.shape[0]))
	temp[0,:] = torch.real(Channel_input)
	temp[1,:] = torch.imag(Channel_input)
	Channel_X = temp
	
	temp = torch.zeros((2,Channel_output.shape[0]))
	temp[0,:] = torch.real(Channel_output)
	temp[1,:] = torch.imag(Channel_output)
	Channel_y = temp

	temp = torch.zeros((2,CFO_input.shape[0]))
	temp[0,:] = torch.real(CFO_input)
	temp[1,:] = torch.imag(CFO_input)
	CFO_X = temp
	CFO_y = CFO_output

	return [RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y]


class TrainDataset(Dataset):
	def __init__(self, file_list, class_ids, args):

		self.file_list = file_list
		self.class_ids = class_ids
		self.args = args
		
		# shuffle the file list
		random.shuffle(self.file_list)

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):

		file_path = self.file_list[index]
		[RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y]  = read_file(file_path)
		
		""" slice only the RF_X """
		slice_index = random.randint(0, RF_X.shape[1] - self.args.slice_len)  # pick a random index from which a slice starts
		RF_X = RF_X[:, slice_index:slice_index+self.args.slice_len]    # pick the slice with determined length and create X (input)
		

		return RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y

if __name__ == '__main__':

	file_path = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/OracleDatasetProcessed-arranged/RFfingerprinting_run1_Radio9_8ft_984.mat' 

	[X1, y1, X2, y2, X3, y3] = read_file(file_path)
	print(X1.shape, y1, X2.shape, y2.shape, X3.shape, y3.shape)
	print(y3)
#torch.Size([2, 4000]) 9 torch.Size([2, 160]) torch.Size([1, 1]) torch.Size([2, 160]) torch.Size([2, 52])