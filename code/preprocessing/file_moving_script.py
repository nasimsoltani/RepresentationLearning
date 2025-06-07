import glob
import os
from tqdm import tqdm

source_dir_base = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/OracleDatasetProcessed-selected'
destination_dir = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/OracleDatasetProcessed-arranged'

for i in range(0,8):
	source_dir = source_dir_base + '/Run1-Radio'+str(2*i)+'-Radio'+str(2*i+1)
	print(source_dir)
	source_file_list = glob.glob(source_dir+'/*.mat')
	print(len(source_file_list))

	for file_path in tqdm(source_file_list):
		filename = file_path.split('/')[-1]
		destination_path = os.path.join(destination_dir , filename)
		os.rename(file_path, destination_path)
