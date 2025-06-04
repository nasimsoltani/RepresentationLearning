from test_model import test_model
from model import IQNet, SpectrogramNet
import os
import torch
import numpy as np
import argparse
import random
import pickle
from ood_detection import id_cluster_former, knn_ood_detector
from retraining_pipeline import create_rehearsal_buffer, retrain_model

from data_generator import IQ2Img_path_convertor

#from rehearsal_buffer import rehearsal_buffer, subtrace_concatenater

if __name__ == "__main__":

	""" This part tests the model on both ID and OOD test samples """
	parser = argparse.ArgumentParser(description = 'Test pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used')
	parser.add_argument('--pkl_dataset_path', default='/home/nasim/', type=str, help='include the path to your ORAN code folder')
	parser.add_argument('--weight_path', default='', type=str, help='include the path to the pt model')
	parser.add_argument('--chop_size', default= 2048, type=int, help='the short example length to chop long examples')
	parser.add_argument('--slice_size', default= 512, type=int, help='the input size of the neural network')
	parser.add_argument('--fft', action='store_true', help='feed fft of input to the NN')
	parser.add_argument('--augmentation', action='store_true', help='augment the input')
	parser.add_argument('--spectrogram', action='store_true', help='augment the input')
	parser.add_argument('--retrain_for_class', default= 'AM', type=str, help='The class you want the NN learn in the next round')

	args = parser.parse_args()

	save_path = '/home/ns38942/AiR/results/' 

	# Initial configurations
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


	#class_ids = {'BPSK':0, 'CW':1, 'FAWGN':2, 'FHCW':3, 'LFM':4, 'LFM2':5, 'OOK':6, 'PCW':7}  #1600 samples in each class
	class_ids = {'AM':0, 'BOCRBPSK':1, 'CW':2, 'FAWGN':3, 'FHCW':4, 'LFM':5, 'PCW':6, 'RBPSK':7, 'ROOK':8}  #1600 samples in each class
    
	OOD_classes_in_weight_filename = args.weight_path.split('OOD_')[-1].rstrip('.pt').split('_')
	OOD_classes = []
	for ood_class in OOD_classes_in_weight_filename:
		OOD_classes.append(ood_class)
	#OOD_classes = ['LFM','LFM2','PCW']
	print(OOD_classes)

	file_suffix = ''
	for this_class in OOD_classes:
		file_suffix += '_'+this_class

	ID_classes = list(filter(lambda x: x not in OOD_classes, list(class_ids.keys())))
	print(ID_classes)
	# classes of interest:
	#all_classes = ['BPSK','CW','FAWGN','FHCW','OOK']
	all_classes = list(class_ids.keys())

	# do the test with desired test dataset and desired trained model
	test_class_map = {}
	class_cntr = 0
	for class_name in ID_classes:
		test_class_map[class_name] = class_cntr
		class_cntr += 1

	with open (args.pkl_dataset_path,'rb') as handle:
		dataset_dict = pickle.load(handle)
	test_list = dataset_dict['test']
	#train_list = dataset_dict['train']

	#random.shuffle(test_list)
	#test_list = test_list[:100]

	num_classes = len(ID_classes)
	if args.spectrogram:
		num_channels = 3
		model = SpectrogramNet(args.slice_size,num_channels,num_classes)
	else:
		num_channels = 2
		model = IQNet(args.slice_size,num_channels,num_classes)
	
	state_dict = torch.load(os.path.join(args.weight_path))
	model.load_state_dict(state_dict)    

	pred_dict = test_model(model, test_list, test_class_map, args, forming_ID_clusters=False)
    
	with open(os.path.join(save_path,'preds'+file_suffix+'.pkl'), 'wb') as handle:
		pickle.dump(pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # form in distribution clustersi
	# get the training list
	
	# do these only if we have any OOD classes
	if len(OOD_classes) != 0:
		train_list = dataset_dict['train']

		
		#random.shuffle(train_list)
		#train_list = train_list[:100]
		
		pred_dict_id = test_model(model, train_list, test_class_map, args, forming_ID_clusters = True)
	
		print('OOD results')





		"""pred_dict_id = in_distribution_clusters(ds_file, norm_param_path, model, all_classes, ID_classes, slice_len)"""

		# KNN OOD detection accuracy

		feature_dimension = 512  #int(2*slice_len) 
		train_set_feature_dict = pred_dict_id['features']
		test_feature_dict = pred_dict['features']
		test_example_dict = pred_dict['examples']
		ood_rate_dict = {}   #{5:[],10:[],15:[]}


		
		# form ID clusters using KNN API

		knn_dict = {}
		for k_neighbor in [5,10,15]:
			print(k_neighbor)	
			knn, center_list, max_distance_list, X_train, y_train, train_set_stride_list = id_cluster_former(train_set_feature_dict, ID_classes, feature_dimension, k_neighbor)
			
			if k_neighbor == 15:
				knn_dict['knn_model'] = knn
				knn_dict['features_X'] = X_train
				knn_dict['features_Y'] = y_train
				knn_dict['cluster_stride'] = train_set_stride_list


			knn_bundle = [knn, X_train, y_train, center_list, max_distance_list, train_set_stride_list] 
			ood_detected_dict, ood_example_dict = knn_ood_detector(knn_bundle, test_feature_dict, ID_classes)
			#print('--------------------------------------')
			#print('k = ' +str(k_neighbour))
			#for i in ood_detection_rate:
			#	print(i)

			ood_rate_dict[k_neighbor] = ood_detected_dict

		with open(os.path.join(save_path,'preds_OOD_results'+file_suffix +'.pkl'), 'wb') as handle:
			pickle.dump(ood_rate_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
		with open('knn.pkl', 'wb') as handle:
			pickle.dump(knn_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	"""
	
		# do another training round:

		print(' start next training round')
		
		# first find the ood_detected
		desired_ood_list = []
		all_ood_datalist = dataset_dict['ood_for_buffer']
		for ood_path in all_ood_datalist:
			this_label = ood_path.split('/')[-2]
			if this_label == args.retrain_for_class:
				IQ_separated = read_file(ood_path)
				desired_ood_list.append(IQ_separated)
		
		# Now pass this list through test model and return only the correct samples:
		pred_dict = test_model(model, desired_ood_list, test_class_map, args.slice_size, args.chop_size, forming_ID_clusters = False)
		dummy_feature_dict = pred_dict['features']
		dummy_example_dict = pred_dict['examples'] 
		# use the knn with k=15
		ood_detected_dict, ood_example_dict = knn_ood_detector(knn, max_distance_list, center_list, X_train, y_train, dummy_feature_dict, dummy_example_dict)

		
		# create the previous_val_dict
		previous_val_dict = {}
		for this_class in ID_classes:
			previous_val_dict[this_class] = []
		for ex_path in dataset_dict['val']:
			IQ_separated = read_file(ex_path)
			this_label = ex_path.split('/')[-2]
			short_IQ_list = signal_chopper(IQ_separated, args.chop_size)
			previous_val_dict[this_label].extend(short_IQ_list)
		

		buffer_train_dict, buffer_val_dict = create_rehearsal_buffer(pred_dict_id['examples'], previous_val_dict, ood_example_dict, 1.0)
		
		# rehearsal buffer is ready, do training
		retrain_model(buffer_train_dict, buffer_val_dict, class_ids, args.slice_len, 256, 2)
	
	"""
	# test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
	"""with open(args.Round+'-ood_rate-'+args.weight_path.split('-')[-1].split('.')[0]+'.pkl', 'wb') as handle:
		pickle.dump(ood_rate_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)"""
