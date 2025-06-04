import torch
import numpy as np
from tqdm import tqdm
import time
import pickle
from scipy import signal

from data_generator import read_file, IQ2Img_path_convertor
import torch.nn.functional as F

from torch.fft import fft as FFT
from torch.fft import fftshift as FFTshift

def signal_chopper(long_signal, chop_size):
	short_signal_list = []
	start_index = 0
	while start_index <= long_signal.shape[1]-chop_size:
		short_signal_list.append(long_signal[:, start_index:start_index+chop_size])
		start_index += chop_size
	return short_signal_list


def batch_creator(args, example, test_stride, file_type):
	if file_type == 'spectrogram':
		data_batch = torch.unsqueeze(example,0)
 
	if file_type == 'iq':
		# slice the IQ_separated
		slice_count = int((example.shape[1]-args.slice_size+1)/test_stride)

		data_batch = torch.zeros((slice_count, 2, args.slice_size))
		start_index = 0
		for i in range (slice_count):
			X = example[:,start_index:start_index+args.slice_size]
			start_index += test_stride		
					
					
			if args.augmentation:
				standard_dev = 1.0/50.0   #1/(2*9*np.sqrt(2))
				this_filter = torch.normal(mean=0.0, std=standard_dev, size = (2,8))
				this_filter = torch.unsqueeze(torch.unsqueeze(this_filter,0),0)
				if ex.split('/')[-2] == 'CW': 
					X = torch.squeeze(torch.squeeze(F.conv2d(torch.unsqueeze(torch.unsqueeze(X,0),0), this_filter, padding='same', groups=1)))
					

			if args.fft:
				complex_X = X[0,:] + 1j*X[1,:]
				complex_X = FFT(complex_X)    
				complex_X = FFTshift(complex_X)
				X[0,:] = torch.real(complex_X)    
				X[1,:] = torch.imag(complex_X)    

				#X = torch.unsqueeze(torch.log10(X[0,:]**2 + X[1,:]**2),0)
	
			#RMS = torch.sqrt(torch.mean(X[0,:]**2 + X[1,:]**2))
			#X  = X/RMS

			data_batch[i,:,:] = X
	
	return data_batch


def test_model(model, test_list, class_ids, args, forming_ID_clusters):
	print('forming ID clusters')
	print(forming_ID_clusters)

	# initialize:
	accuracy = 0
	Conf = np.zeros((len(class_ids),len(class_ids)))
	test_stride = 10
	pred_dict = {}

	if args.spectrogram:
		file_type = 'spectrogram'
		test_list = IQ2Img_path_convertor(test_list)
	else:
		file_type = 'iq'

	print('length of test list is: ' +str(len(test_list)))

	# We don't need gradients on to do reporting
	model.train(False)
	model.cuda()
	# model.cpu()
	model.eval()

	# prepare the feature_dict 
	feature_dict, example_dict, preds = {}, {}, {}
	num_classes = len(list(class_ids.keys()))

	# run the test on the test dataset
	with torch.no_grad():
		correct_cntr =  torch.zeros((num_classes))
		total_cntr = torch.zeros((num_classes))
		
		runtime_list = []

		for ex in tqdm(test_list):
			file_content = read_file(ex,file_type)
			
			# New example, so make the key in the dictionary
			preds[ex] = []			

	
			if file_type == 'iq':
				# chop to shorter sizes:
				short_signal_list = signal_chopper(file_content, args.chop_size)
			else:
				short_signal_list = [file_content] 
			
		
			for example in short_signal_list:

				data_batch = batch_creator(args, example, test_stride, file_type)

				# pass through model
				start_time = time.time()	
				data_batch = data_batch.cuda().float()
				feature_hat, y_hat = model(data_batch)

				#print(feature_hat.shape, y_hat.shape)
				end_time = time.time()
				runtime_list.append(end_time - start_time)

				# rewriting the original feature_hat and y_hat because we do not need it anymore
				feature_hat = torch.sum(feature_hat, axis=0)
				y_hat = torch.unsqueeze(torch.sum(y_hat, axis=0), dim=0)

				# normalize features
				#RMS =  torch.sqrt(torch.mean(feature_hat**2)) 
				#feature_hat = feature_hat/RMS
			
				# now y_hat and feature_hat are for just one short example

				# if we are forming ID clusters, we do not need to know the corresponding example path
				# for each input signal, so the dict_key is just the class
				# if we are just testing on a "test" set, we like to know the example path
				# to be able to extract SNR later and report results based on SNR
				if forming_ID_clusters:
					dict_key = ex.split('/')[-2]
				else:
					dict_key = ex				

				if dict_key not in list(feature_dict.keys()):
				# create that key, put an empty list there, and then append
					feature_dict[dict_key] = []
					#example_dict[dict_key] = []
				
				if not forming_ID_clusters:
					feature_dict[ex].append(feature_hat.cpu()) 
					#example_dict[ex].append(example.cpu()) 

				# save prediction for each example and associate it with each example
				preds[ex].append(y_hat.cpu())
			
				# only if the test signal is from an ID class do the following
				if ex.split('/')[-2] in list(class_ids.keys()):
				
 
					# prepare label
					true_label = class_ids[ex.split('/')[-2]]
					#true_label_batch = torch.ones((slice_count,1))*true_label

					if forming_ID_clusters: 
					# choose only the correct features
						pred = torch.argmax(y_hat.cpu(),dim=-1)
						#if pred == true_label:
						feature_dict[ex.split('/')[-2]].append(feature_hat.cpu()) 
						#example_dict[ex.split('/')[-2]].append(example.cpu()) 


	
					for pred in torch.argmax(y_hat.cpu(),dim=-1):
						Conf[true_label,pred] += 1

					pred = torch.argmax(y_hat.cpu(),dim=-1)
	
					correct_cntr[true_label] += torch.sum(torch.argmax(y_hat.cpu(),dim=-1) == true_label)
					total_cntr[true_label] += y_hat.shape[0]


	# normalize conf before sending out:
	norm_conf = np.zeros(Conf.shape)
	for i in range(Conf.shape[0]):
		for j in range(Conf.shape[1]): # for each predicted
			norm_conf[i,j] = (Conf[i,j]/np.sum(Conf[i,:]))*100
	Conf = norm_conf

	with open('runtime.pkl','wb') as handle:
		pickle.dump({'run_time':runtime_list},handle)

	accuracy = correct_cntr/total_cntr
	print('correct count: ' +str(correct_cntr) + ' and total count: ' +str(total_cntr))
	print('Accuracy is: '+str(accuracy))

	pred_dict = {'features':feature_dict, 'confusion_matrix':Conf, 'examples':example_dict, 'predictions':preds} 

	return pred_dict
