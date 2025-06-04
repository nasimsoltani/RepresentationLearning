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
#from PIL import Image, ImageChops
import cv2
import glob

def read_file(file_path, file_type):
	if file_type == 'iq':
		IQ = np.fromfile(file_path, dtype=np.complex64)  #length = 16384
		IQ = torch.from_numpy(IQ)
		# reshape the IQ sequence and remove the extra axis
		# calculate RMS and normalize the IQ sequence
		RMS = torch.sqrt(torch.mean(torch.real(IQ)**2+torch.imag(IQ)**2))
		IQ = IQ/RMS
		# separate I and Q as two separate channels and bring the channels
		# to the first dimension for pytorch compatibility

		IQ_separated = torch.zeros((2,IQ.shape[0]))
		IQ_separated[0,:] = torch.real(IQ)
		IQ_separated[1,:] = torch.imag(IQ)
		X = IQ_separated
	else:
		#this_folder = file_path.split('/')[-2]
		#filename = file_path.split('/')[-1].split('.')[0]
		#new_file_path = os.path.join(os.path.abspath('cropped_images'), this_folder, str(filename)+'.jpg') 
		X = cv2.imread(file_path)          # reading the image
		
		#X = Image.open(new_file_path)	
		# normalize, resize, and reshape the image
		X = X/255.
		X = cv2.resize(X, (300,200))         # dimensions must be (width, height)
		#X = X.resize((160, 90))
		X = np.moveaxis(X, -1, 0)           # reshape to channel first	
		X = torch.from_numpy(X)

	return X

def IQ2Img_path_convertor(IQ_source_list):
	image_base = os.path.abspath('cropped_images')
	image_list = []
	print('******** converting IQ file paths to spectrogram paths *******')
	for filepath in tqdm(IQ_source_list):
		filename = filepath.split('/')[-1].split('.')[0]
		foldername = filepath.split('/')[-2]
		image_path = os.path.join(image_base,foldername,filename)
		image_list.extend(glob.glob(image_path+'*'))
		
	return image_list

class TrainDataset(Dataset):
	def __init__(self, IQ_file_list, labels, class_ids, args, slice_size, rehearsal_buffer, round_index):

		self.IQ_file_list = IQ_file_list
		self.labels = labels
		self.class_ids = class_ids
		self.slice_size = slice_size
		self.args = args
		
		if args.spectrogram:
			self.file_type = 'spectrogram'	
			self.IQ_file_list = IQ2Img_path_convertor(self.IQ_file_list)
		else:
			self.file_type = 'iq'	
	
		if round_index == 1:
			self.data_cache = {}

			# create keys (classes) for data_cache
			for class_name in list(self.class_ids.keys()):
				self.data_cache[class_name] = []

			# load all data to cache:
			print('Adding all files to cache')
			for IQ_path in tqdm(self.IQ_file_list):
				#this_class = self.labels[IQ_path]
				this_class = IQ_path.split('/')[-2]	
				self.__add_to_cache(IQ_path, this_class)
		else:
			self.data_cache = rehearsal_buffer

		print('len of IQ_file_list and data_cache: ')
		print(len(self.IQ_file_list), len(self.data_cache.keys()))

	def __len__(self):
	# loop over the class as many as we define here
		return len(self.data_cache[list(self.data_cache.keys())[0]])*100*len(list(self.data_cache.keys()))

	def __add_to_cache(self, file_path, this_class):
		X = read_file(file_path, self.file_type)
		self.data_cache[this_class].append(X)

	def __getitem__(self, index):

		#Generate two samples of data (anchor and positive)
		this_random_class = random.sample(list(self.data_cache.keys()), 1)[0]
		anchor_and_positive = random.sample(self.data_cache[this_random_class], 2)
		X = anchor_and_positive[0]
		""" No slicing """
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
		
		""" No slicing """
		slice_index = random.randint(0, negative_X.shape[1] - self.slice_size)  # pick a random index from which a slice starts
		negative_X = negative_X[:, slice_index:slice_index+self.slice_size]    # pick the slice with determined length and create X (input)
		

		## augmentation ##


		if self.args.augmentation: 
			standard_dev = 1.0/50.0   #1/(2*9*np.sqrt(2))
			this_filter = torch.normal(mean=0.0, std=standard_dev, size = (2,8))
			this_filter = torch.unsqueeze(torch.unsqueeze(this_filter,0),0)
			#if this_random_class == 'CW': 
			X = torch.squeeze(torch.squeeze(F.conv2d(torch.unsqueeze(torch.unsqueeze(X,0),0), this_filter, padding='same', groups=1)))
			positive_X = torch.squeeze(torch.squeeze(F.conv2d(torch.unsqueeze(torch.unsqueeze(positive_X,0),0), this_filter, padding='same', groups=1)))
		
			#if negative_class == 'CW': 
			negative_X = torch.squeeze(torch.squeeze(F.conv2d(torch.unsqueeze(torch.unsqueeze(negative_X,0),0), this_filter, padding='same', groups=1)))



		if self.args.fft:
			# Do FFT
			# convert to complex
			complex_X = X[0,:] + 1j*X[1,:]
			complex_negative_X = negative_X[0,:] + 1j*negative_X[1,:]
			complex_positive_X = positive_X[0,:] + 1j*positive_X[1,:]
		
			complex_X = FFT(complex_X)		
			complex_positive_X = FFT(complex_positive_X)		
			complex_negative_X = FFT(complex_negative_X)		

			complex_X = FFTshift(complex_X)
			complex_positive_X = FFTshift(complex_positive_X)
			complex_negative_X = FFTshift(complex_negative_X)
	
	

			X[0,:] = torch.real(complex_X)			
			X[1,:] = torch.imag(complex_X)			
			positive_X[0,:] = torch.real(complex_positive_X)			
			positive_X[1,:] = torch.imag(complex_positive_X)			
			negative_X[0,:] = torch.real(complex_negative_X)			
			negative_X[1,:] = torch.imag(complex_negative_X)			

		
		#if self.args.mask:
			
	

	
		"""RMS = torch.sqrt(torch.mean(X[0,:]**2 + X[1,:]**2))
		X  = X/RMS
		RMS = torch.sqrt(torch.mean(positive_X[0,:]**2 + positive_X[1,:]**2))
		positive_X = positive_X/RMS
		RMS = torch.sqrt(torch.mean(negative_X[0,:]**2 + negative_X[1,:]**2))
		negative_X = negative_X/RMS"""






		return X, positive_X, negative_X, y

class SpectrogramTrainDataset(TrainDataset):
	def __len__(self):
		# loop over the class as many as we define here
		return len(self.data_cache[list(self.data_cache.keys())[0]])*len(list(self.data_cache.keys()))


	def __getitem__(self, index):
		#Generate two samples of data (anchor and positive)
		this_random_class = random.sample(list(self.data_cache.keys()), 1)[0]
		anchor_and_positive = random.sample(self.data_cache[this_random_class], 2)
		X = anchor_and_positive[0]

		positive_X = anchor_and_positive[1]
		y = int(self.class_ids[this_random_class])

		## load the negative sample
		negative_class_list = list(self.data_cache.keys())
		negative_class_list.remove(this_random_class)
		negative_class = random.sample(negative_class_list, 1)[0]
		negative_X = random.sample(self.data_cache[negative_class], 1)[0]

		return X, positive_X, negative_X, y

# class TestDataset(TrainDataset):
#     # over-writing the __getitem__ function in the parent class
#     def __getitem__(self, index):
#         IQ_path = list(self.data_cache.keys())[index]
#         X = self.data_cache[list(self.data_cache.keys())[index]]       # creating X (input)
#         y = int(self.class_ids[self.labels[list(self.data_cache.keys())[index]]])   # creating y (output)

#         return X, y, IQ_path

