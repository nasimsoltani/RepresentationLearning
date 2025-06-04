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
import cv2
import glob
from scipy.io import loadmat 

def read_file(file_path):
	""" gets a file_path for RF fingerprinting input part, and return associated data parts
	for CFO estimation and channel estimation too """

	print(file_path)

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


	print(RF_input.shape, CFO_input.shape, Channel_input.shape)
	print(Channel_output.shape)

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
	def __init__(self, IQ_file_list, labels, class_ids, args, slice_size, rehearsal_buffer, round_index):

		self.IQ_file_list = IQ_file_list
		self.labels = labels
		self.class_ids = class_ids
		self.slice_size = slice_size
		self.args = args
		
	
		if round_index == 1:
			self.data_cache = {}

			# create keys (classes) for data_cache
			for class_name in list(self.class_ids.keys()):
				self.data_cache[class_name] = []

			# load all data to cache:
			print('Adding all files to cache')
			for IQ_path in tqdm(self.IQ_file_list):
				this_class = IQ_path.split('/')[-1].split('_')[-3]	
				self.__add_to_cache(IQ_path, this_class)
		else:
			self.data_cache = rehearsal_buffer

		print('len of IQ_file_list and data_cache: ')
		print(len(self.IQ_file_list), len(self.data_cache.keys()))

	def __len__(self):
	# loop over the class as many as we define here
		return len(self.data_cache[list(self.data_cache.keys())[0]])*100*len(list(self.data_cache.keys()))

	def __add_to_cache(self, file_path, this_class):
		Xy_bundle = read_file(file_path, self.file_type)
		self.data_cache[this_class].append(Xy_bundle)

	def __getitem__(self, index):

		#Generate two samples of data (anchor and positive)
		this_random_class = random.sample(list(self.data_cache.keys()), 1)[0]
		
		RF_X, RF_y,  = random.sample(self.data_cache[this_random_class], 1)[0]
		
		""" slice only the RF_X """
		slice_index = random.randint(0, X.shape[1] - self.slice_size)  # pick a random index from which a slice starts
		X = X[:, slice_index:slice_index+self.slice_size]    # pick the slice with determined length and create X (input)
		positive_X = anchor_and_positive[1]
		slice_index = random.randint(0, positive_X.shape[1] - self.slice_size)  # pick a random index from which a slice starts
		positive_X = positive_X[:, slice_index:slice_index+self.slice_size]    # pick the slice with determined length and create X (input)
		y = int(self.class_ids[this_random_class])

		## load the negative sample
		negative_class_list = list(self.data_cache.keys())
		negative_class_list.remove(this_random_class)
		negative_class = random.sample(negative_class_list, 1)[0]
		negative_X = random.sample(self.data_cache[negative_class], 1)[0]
		
		""" slicing """
		slice_index = random.randint(0, negative_X.shape[1] - self.slice_size)  # pick a random index from which a slice starts
		negative_X = negative_X[:, slice_index:slice_index+self.slice_size]    # pick the slice with determined length and create X (input)
		



		return X, positive_X, negative_X, y

if __name__ == '__main__':

	file_path = '/home/ns38942/RepresentationLearning/dataset/RFfingerprinting_run1_Radio9_8ft_984.mat' 

	[X1, y1, X2, y2, X3, y3] = read_file(file_path)
	print(X1.shape, y1, X2.shape, y2.shape, X3.shape, y3.shape)
