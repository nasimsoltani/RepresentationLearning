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

def read_file(file_path, max_cfo):
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
	CFO_output = torch.from_numpy(content['CFO']).float().squeeze()  # Fix dtype and shape
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
	def __init__(self, file_list, class_ids, args, max_cfo, mean_cfo, std_cfo, test_mode=False):

		self.file_list = file_list
		self.class_ids = class_ids
		self.args = args
		self.max_cfo = max_cfo
		self.mean_cfo = mean_cfo
		self.std_cfo = std_cfo
		self.test_mode = test_mode
		
		# shuffle the file list
		if not self.test_mode:
			random.shuffle(self.file_list)

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):

		try:

			file_path = self.file_list[index]
			[RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y]  = read_file(file_path, self.max_cfo)
			
			if self.test_mode:
				# In test mode, we create sliding window slices of RF_X for evaluation.
				if RF_X.shape[1] >= self.args.slice_len:
					# Use unfold for efficient slicing
					RF_X = RF_X.unfold(1, self.args.slice_len, 1).permute(1, 0, 2)
				else:
					# Pad if the sequence is shorter than slice_len
					padded_RF_X = torch.zeros((RF_X.shape[0], self.args.slice_len))
					padded_RF_X[:, :RF_X.shape[1]] = RF_X
					RF_X = padded_RF_X.unsqueeze(0)
			else:
				""" slice only the RF_X """
				slice_index = random.randint(0, RF_X.shape[1] - self.args.slice_len)  # pick a random index from which a slice starts
				RF_X = RF_X[:, slice_index:slice_index+self.args.slice_len]    # pick the slice with determined length and create X (input)

			#print(RF_X.shape, RF_y, CFO_y.shape, Channel_X.shape, Channel_y.shape)
			
			CFO_y = ((CFO_y - self.mean_cfo)/self.std_cfo).float()  # Ensure float dtype and proper shape
			#print(CFO_y)

			return RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y, file_path

		except Exception as e:
			print(f"Error: {e}")
			file_path = self.file_list[0]
			[RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y]  = read_file(file_path, self.max_cfo)
			
			if self.test_mode:
				# In test mode, we create sliding window slices of RF_X for evaluation.
				if RF_X.shape[1] >= self.args.slice_len:
					# Use unfold for efficient slicing
					RF_X = RF_X.unfold(1, self.args.slice_len, 1).permute(1, 0, 2)
				else:
					# Pad if the sequence is shorter than slice_len
					padded_RF_X = torch.zeros((RF_X.shape[0], self.args.slice_len))
					padded_RF_X[:, :RF_X.shape[1]] = RF_X
					RF_X = padded_RF_X.unsqueeze(0)
			else:
				""" slice only the RF_X """
				slice_index = random.randint(0, RF_X.shape[1] - self.args.slice_len)  # pick a random index from which a slice starts
				RF_X = RF_X[:, slice_index:slice_index+self.args.slice_len]    # pick the slice with determined length and create X (input)

			#print(RF_X.shape, RF_y, CFO_y.shape, Channel_X.shape, Channel_y.shape)

			#CFO_y = CFO_y/self.max_cfo
			
			CFO_y = ((CFO_y - self.mean_cfo)/self.std_cfo).float()  # Ensure float dtype and proper shape
			#print(CFO_y)


			return RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y, file_path



class TrainDatasetRFixed(Dataset):
	def __init__(self, file_list, class_ids, args, max_cfo, mean_cfo, std_cfo, test_mode=False,rf_begin_idx=0):

		self.file_list = file_list
		self.class_ids = class_ids
		self.args = args
		self.max_cfo = max_cfo
		self.mean_cfo = mean_cfo
		self.std_cfo = std_cfo
		self.test_mode = test_mode
		self.rf_begin_idx = rf_begin_idx
		
		# shuffle the file list
		if not self.test_mode:
			random.shuffle(self.file_list)

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):

		try:

			file_path = self.file_list[index]
			[RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y]  = read_file(file_path, self.max_cfo)

			RF_X = RF_X[:, 400:]
			
			slice_len = 1024
			slices = []
			for i in range(3):
				start_idx = i * slice_len
				end_idx = start_idx + slice_len
				
				if start_idx >= RF_X.shape[1]:
					break

				rf_slice = RF_X[:, start_idx:end_idx]

				if rf_slice.shape[1] < slice_len:
					padding_size = slice_len - rf_slice.shape[1]
					rf_slice = F.pad(rf_slice, (0, padding_size), "constant", 0)

				slices.append(rf_slice)
			
			if not slices:
				# Create a dummy slice to avoid errors, this sample will be effectively skipped
				RF_X = torch.zeros((0, 2, slice_len))
			else:
				RF_X = torch.stack(slices)

			CFO_y = ((CFO_y - self.mean_cfo)/self.std_cfo).float()

			return RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y, file_path

		except Exception as e:
			print(f"Error processing file {self.file_list[index]}: {e}")
			# Return a dummy sample
			slice_len = 1024
			RF_X = torch.zeros((0, 2, slice_len))
			RF_y = -1 
			CFO_X = torch.zeros((2, 160))
			CFO_y = torch.tensor(0.0)
			Channel_X = torch.zeros((2, 160))
			Channel_y = torch.zeros((2, 52))
			return RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y, self.file_list[index]


def collate_fn_rf_fixed(batch):
    # Filter out samples with no RF_X slices
    batch = [b for b in batch if b[0].shape[0] > 0]
    if not batch:
        # If batch is empty after filtering, return None. Dataloader loop should handle this.
        return None, None, None, None, None, None, None

    rf_x_list, rf_y_list, cfo_x_list, cfo_y_list, channel_x_list, channel_y_list, paths = [], [], [], [], [], [], []

    for sample in batch:
        rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, path = sample
        num_slices = rf_x.shape[0]

        rf_x_list.append(rf_x)
        rf_y_list.extend([rf_y] * num_slices)
        cfo_x_list.extend([cfo_x] * num_slices)
        cfo_y_list.extend([cfo_y] * num_slices)
        channel_x_list.extend([channel_x] * num_slices)
        channel_y_list.extend([channel_y] * num_slices)
        paths.append(path)

    # Convert lists to tensors
    final_rf_x = torch.cat(rf_x_list, dim=0)
    final_rf_y = torch.tensor(rf_y_list, dtype=torch.long)
    final_cfo_x = torch.stack(cfo_x_list)
    final_cfo_y = torch.stack(cfo_y_list)
    final_channel_x = torch.stack(channel_x_list)
    final_channel_y = torch.stack(channel_y_list)

    return final_rf_x, final_rf_y, final_cfo_x, final_cfo_y, final_channel_x, final_channel_y, paths
