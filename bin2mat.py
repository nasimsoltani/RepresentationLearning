import numpy as np
import os
import glob
from scipy.io import savemat
from tqdm import tqdm

source_path = '/home/ns38942/Oracle/KRI-16Devices-RawData/' 
destination_path = '/home/ns38942/Oracle/mat_dataset/'

# {'3123D78': 0, '3123D52': 1, '3123D80': 2, '3123D7D': 3, '3123D76': 4, '3123D89': 5, '3123D54': 6, '3123D7E': 7, '3123D65': 8, '3123D79': 9, '3123D70': 10, '3123D7B': 11, '3124E4A': 12, '3123D58': 13, '3123D64': 14, '3123EFE': 15}


file_paths = glob.glob(source_path + '*/*.sigmf-data')
#file_paths = ['/home/ns38942/Oracle/KRI-16Devices-RawData/2ft/WiFi_air_X310_3123D52_2ft_run2.sigmf-data']

# creating the device_id_map dictionary
device_id_map = {}
device_id = 0

for file_path in file_paths:

	device = file_path.split('/')[-1].split('.')[0].split('_')[-3]
	if device not in device_id_map:
		device_id_map[device] = device_id
		device_id = device_id + 1

print(device_id_map)

for file_path in tqdm(file_paths):
	
	IQ = np.fromfile(file_path, dtype=np.complex128)
	#print(IQ[:100])
	#print(IQ.shape)
	### try this 
	#iq = np.fromfile(file_path, dtype=np.float64)
	#print(iq[:200])
	#complex_iq = np.zeros(

	device = file_path.split('/')[-1].split('.')[0].split('_')[-3]
	run = file_path.split('/')[-1].split('.')[0].split('_')[-1]
	distance = file_path.split('/')[-1].split('.')[0].split('_')[-2]
	
	filename = run + '_Radio'+ str(device_id_map[device]) + '_' + distance + '.mat'
	path = destination_path + filename
	this_dict = {'Rx':IQ}

	savemat(path , this_dict)

