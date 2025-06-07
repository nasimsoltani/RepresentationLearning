import glob
import random
import pickle

base_path = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/OracleDatasetProcessed-arranged'
portion_to_use = 0.5
rf_pkl_path = f'/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_{portion_to_use}.pkl'
cfo_pkl_path = f'/home/hofmann/Documents/projects/RepresentationLearning/dataset/cfo_partition_dict_{portion_to_use}.pkl'
channel_pkl_path = f'/home/hofmann/Documents/projects/RepresentationLearning/dataset/channel_partition_dict_{portion_to_use}.pkl'

# create Radio list
radio_list = list(range(0,16))
radio_list = list(map(lambda x: 'Radio'+str(x), radio_list))

print(radio_list)

# create distance list
distance_list = list(range(2,62,6))
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

with open (cfo_pkl_path, 'wb') as handle:
	pickle.dump(cfo_partition_dict, handle)

with open (channel_pkl_path, 'wb') as handle:
	pickle.dump(channel_partition_dict, handle)