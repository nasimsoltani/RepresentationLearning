import glob
import random
import pickle
from scipy.io import loadmat
from dotenv import load_dotenv
import os
import numpy as np

# Load environment variables
load_dotenv()

# Get portion_to_use from environment variable, default to 0.5 if not set
portion_to_use = float(os.getenv('PORTION_TO_USE', '0.5'))

"""base_path = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/OracleDatasetProcessed-arranged'
rf_pkl_path = f'/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_{portion_to_use}.pkl'
cfo_pkl_path = f'/home/hofmann/Documents/projects/RepresentationLearning/dataset/cfo_partition_dict_{portion_to_use}.pkl'
channel_pkl_path = f'/home/hofmann/Documents/projects/RepresentationLearning/dataset/channel_partition_dict_{portion_to_use}.pkl'
"""

base_path = os.getenv('DATA_BASE_PATH')
if not base_path:
    raise ValueError("Missing DATA_BASE_PATH environment variable. Please check your .env file.")

# Construct paths using os.path.join
dataset_dir = os.path.dirname(base_path)  # Get parent directory of base_path
rf_pkl_path = os.path.join(dataset_dir, f'rf_partition_dict_{portion_to_use}.pkl')
cfo_pkl_path = os.path.join(dataset_dir, f'cfo_partition_dict_{portion_to_use}.pkl')
channel_pkl_path = os.path.join(dataset_dir, f'channel_partition_dict_{portion_to_use}.pkl')

def read_file(file_path):
	""" gets a file_path for RF fingerprinting input part, and returns associated estimated CFO """

	# file_path is the RFfingerprinting file path, read CFO files

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
	CFO_output = content['CFO'][0,0]


	return CFO_output

# create Radio list
radio_list = list(range(0,16))
radio_list = list(map(lambda x: 'Radio'+str(x), radio_list))

print(radio_list)

# create distance list
distance_list = list(range(2,64,6))
distance_list = list(map(lambda x: str(x)+'ft', distance_list))

print(distance_list)

# initialize train/val/test sets for 3 tasks

RF_train_list, RF_val_list, RF_test_list = [], [], []
CFO_train_list, CFO_val_list, CFO_test_list = [], [], []
Channel_train_list, Channel_val_list, Channel_test_list = [], [], []

# Two for loops: (i) over radios, (ii) over distances
for radio in radio_list:
	for distance in distance_list:
		file_list = glob.glob(base_path + '/RFfingerprinting_run1_' + radio + '_' + distance + '_' + '*')
		#print(base_path + 'RFfingerprinting_run1_' + radio + '_' + distance + '_' + '*')
		print(len(file_list))

		# how many files we have for this radio and distance
		num_files = len(file_list)
		# create a list of the file indexes
		index_list = list(range(0,num_files))
		# convert the list to strings
		index_list = list(map(lambda x: str(x), index_list))
		random.shuffle(index_list)

		# use only the portion of files that is specified
		index_list = index_list[:int(portion_to_use*len(index_list))]

		train_index_list = index_list[:int(0.7*len(index_list))]
		val_index_list = index_list[int(0.7*len(index_list)):int(0.8*len(index_list))]
		test_index_list = index_list[int(0.8*len(index_list)):]
		
		# rf fingerprinting partitions
		RF_train_list.extend(list(map(lambda x: base_path + '/RFfingerprinting_run1_' + radio + '_' + distance + '_' + x + '.mat', train_index_list)))
		RF_val_list.extend(list(map(lambda x: base_path + '/RFfingerprinting_run1_' + radio + '_' + distance + '_' + x + '.mat', val_index_list)))
		RF_test_list.extend(list(map(lambda x: base_path + '/RFfingerprinting_run1_' + radio + '_' + distance + '_' + x + '.mat', test_index_list)))

		# CFO estimation partitions
		CFO_train_list.extend(list(map(lambda x: base_path + '/CFOEstimation_run1_' + radio + '_' + distance + '_' + x + '.mat', train_index_list)))
		CFO_val_list.extend(list(map(lambda x: base_path + '/CFOEstimation_run1_' + radio + '_' + distance + '_' + x + '.mat', val_index_list)))
		CFO_test_list.extend(list(map(lambda x: base_path + '/CFOEstimation_run1_' + radio + '_' + distance + '_' + x + '.mat', test_index_list)))

		# Channel estimation partitions
		Channel_train_list.extend(list(map(lambda x: base_path + '/ChannelEstimation_run1_' + radio + '_' + distance + '_' + x + '.mat', train_index_list)))
		Channel_val_list.extend(list(map(lambda x: base_path + '/ChannelEstimation_run1_' + radio + '_' + distance + '_' + x + '.mat', val_index_list)))
		Channel_test_list.extend(list(map(lambda x: base_path + '/ChannelEstimation_run1_' + radio + '_' + distance + '_' + x + '.mat', test_index_list)))


print(len(RF_train_list))

# now train/val/test partitions for all tasks are ready, dump them in a pickle file
"""RF_temp_dict = { 'train' : RF_train_list, 'val' : RF_val_list, 'test' : RF_test_list }
CFO_temp_dict = { 'train' : CFO_train_list, 'val' : CFO_val_list, 'test' : CFO_test_list }
Channel_temp_dict = { 'train' : Channel_train_list, 'val' : Channel_val_list, 'test' : Channel_test_list }

partition_dict = {'RF_Fingerprinting':RF_temp_dict, 
				'Channel_Estimation':Channel_temp_dict,
				'CFO_Estimation':CFO_temp_dict  }
"""

rf_partition_dict = {'train': RF_train_list, 'val': RF_val_list, 'test': RF_test_list}
cfo_partition_dict = {'train': CFO_train_list, 'val': CFO_val_list, 'test': CFO_test_list}
channel_partition_dict = {'train': Channel_train_list, 'val': Channel_val_list, 'test': Channel_test_list}

with open (rf_pkl_path, 'wb') as handle:
	pickle.dump(rf_partition_dict, handle)

"""with open (cfo_pkl_path, 'wb') as handle:
	pickle.dump(cfo_partition_dict, handle)

with open (channel_pkl_path, 'wb') as handle:
	pickle.dump(channel_partition_dict, handle)
"""

with open (rf_pkl_path, 'rb') as handle:
	partitions = pickle.load(handle)

# max_cfo = 0
# train_list = partitions['train']
# for file_path in train_list:
# 	output_cfo = read_file(file_path)
# 	max_cfo = max(max_cfo , abs(output_cfo))


#calculate mean and std of CFO
cfo_list = []
train_list = partitions['train']
for file_path in train_list:
	output_cfo = read_file(file_path)
	cfo_list.append(output_cfo)

mean_cfo = np.mean(cfo_list)
std_cfo = np.std(cfo_list)
max_cfo = np.max(np.abs(cfo_list))

partitions['mean_cfo'] = mean_cfo
partitions['std_cfo'] = std_cfo
partitions['max_cfo'] = max_cfo
with open (rf_pkl_path, 'wb') as handle:
	pickle.dump(partitions,handle)


