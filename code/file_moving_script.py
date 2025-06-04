import glob
import os
from tqdm import tqdm

destination_dir = '/home/ns38942/RepresentationLearning/dataset/'
source_dir_base = '/home/ns38942/RepresentationLearning/'

for i in range(0,8):
	source_dir = source_dir_base + 'Run1-Radio'+str(2*i)+'-Radio'+str(2*i+1)
	source_file_list = glob.glob(source_dir+'/*.mat')

	for file_path in tqdm(source_file_list):
		filename = file_path.split('/')[-1]
		destination_path = os.path.join(destination_dir , filename)
		os.rename(file_path, destination_path)
