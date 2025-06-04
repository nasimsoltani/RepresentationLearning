import pickle
import numpy as np
import torch
import os


def train_model(model, train_dl, val_dl, epochs, save_path, early_stopping, patience):

	learning_rate = 0.0001
	min_learning_rate = 0.000001
	learning_rate_list = []
	# model.load_state_dict(torch.load('results/weights.pt'))

	model.train()
	model.cuda()

	loss_func1 = torch.jit.script(TripletLoss())
	loss_func2 = torch.nn.CrossEntropyLoss()

	# loss_func = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


	# Now do conventional training
	# train the model
	best_vloss = 1000000
	epoch_patience_cntr = 0
	previous_epoch_vloss = 100000

	train_loss_list, train_loss1_list, train_loss2_list, val_loss_list, val_loss1_list, val_loss2_list = [], [], [], [], [], []
	train_acc_list, val_acc_list = [], []

	for epoch in range(epochs):  # First for loop: loop over the dataset multiple times (epochs)

		print('epoch ' + str(epoch+1) + '/' + str(epochs))

		learning_rate_list.append(learning_rate)

		running_loss = 0.0
		last_loss = 0.0
		all_batches_acc = 0
		all_batches_loss, all_batches_loss1, all_batches_loss2 = 0, 0, 0

		for batch_cntr, (anchor_input, positive_input, negative_input, labels) in enumerate(train_dl):   # Second for loop: in each epoch, loop over batches

			# Every data instance is an input + label pair
			anchor_input = anchor_input.float().cuda()
			positive_input = positive_input.float().cuda()
			negative_input = negative_input.float().cuda()
			labels = labels.long().cuda()

			# pass training batch through the model
			anchor_outputs, label_hat = model(anchor_input)
			positive_outputs, _ = model(positive_input)
			negative_outputs, _ = model(negative_input)


			# Zero your gradients for every batch
			optimizer.zero_grad()
			# compute loss and do the backward path
			# triplet loss
			loss1 = loss_func1(anchor_outputs, positive_outputs, negative_outputs)
			loss2 = loss_func2(label_hat, labels)
			loss = loss1 + loss2
			#loss = loss2  # just classification
			loss.backward()
			# Adjust learning weights
			optimizer.step()

			# now that one round of backward is done, test the model on this training batch to calculated training loss and accuracy
			# passing the batch through the model
			anchor_outputs, label_hat = model(anchor_input)    
			positive_outputs, _ = model(positive_input)
			negative_outputs, _ = model(negative_input)
			# triplet loss:



			loss1 = loss_func1(anchor_outputs, positive_outputs, negative_outputs)
			loss2 = loss_func2(label_hat, labels)
			loss = loss1 + loss2
			#loss = loss2  # just classification
			# compute batch accuracy, and add it to all_batches_acc:
			all_batches_loss1 += loss1.item()
			all_batches_loss2 += loss2.item()
			all_batches_loss += loss.item()

			# Gather loss for reporting just now (running)
			running_loss += loss.item()

			all_batches_acc += torch.mean((torch.argmax(label_hat, dim=-1) == labels).float()).data.cpu().numpy()

			if batch_cntr % 1000 == 999:   # print loss every 1000 batches to avoid over-crowding the logs
				last_loss = running_loss / 1000 # loss per batch
				print('  batch {} loss: {}'.format(batch_cntr + 1, last_loss))
				running_loss = 0.

		# training for one epoch is done over all batches
		# Now calculate training set loss and accuracy
		avg_loss1 = all_batches_loss1/(batch_cntr+1)
		avg_loss2 = all_batches_loss2/(batch_cntr+1)
		avg_loss = all_batches_loss/(batch_cntr+1)
		avg_acc = all_batches_acc/(batch_cntr+1)

		# training for one epoch (all batches) is done, now do the validation
		# put the model in evaluation (test) mode
		model.eval()

		with torch.no_grad():     # We don't need gradients on to do reporting

			running_vloss, running_vloss1, running_vloss2 = 0.0, 0, 0
			all_batches_v_acc = 0

			for batch_cntr, vdata in enumerate(val_dl):
				vinputs_anchor, vinputs_positive, vinputs_negative, vlabels = vdata
				vinputs_anchor = vinputs_anchor.float().cuda()
				vinputs_positive = vinputs_positive.float().cuda()
				vinputs_negative = vinputs_negative.float().cuda()
				vlabels = vlabels.long().cuda()

				voutputs_anchor, vlabel_hats = model(vinputs_anchor)
				voutputs_positive, _ = model(vinputs_positive)
				voutputs_negative, _ = model(vinputs_negative)

				# triplet loss
				vloss1 = loss_func1(voutputs_anchor, voutputs_positive, voutputs_negative)
				vloss2 = loss_func2(vlabel_hats, vlabels)
				vloss = vloss1 + vloss2
				#vloss = vloss2   #just classification
				running_vloss1 += vloss1
				running_vloss2 += vloss2
				running_vloss += vloss

				# calculate validation accuracy:
				all_batches_v_acc += torch.mean((torch.argmax(vlabel_hats, dim=-1) == vlabels).float()).data.cpu().numpy()

			avg_vloss1 = running_vloss1 / (batch_cntr+1)
			avg_vloss2 = running_vloss2 / (batch_cntr+1)
			avg_vloss = running_vloss / (batch_cntr+1)
			print('********* validation Loss = '+str(avg_vloss)+' ************')
			print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
			train_loss1_list.append(avg_loss1)
			train_loss2_list.append(avg_loss2)
			train_loss_list.append(avg_loss)
			val_loss1_list.append(avg_vloss1.cpu().detach().numpy())
			val_loss2_list.append(avg_vloss2.cpu().detach().numpy())
			val_loss_list.append(avg_vloss.cpu().detach().numpy())

			avg_vacc = all_batches_v_acc/(batch_cntr+1)
			print('********* validation accuracy = '+str(avg_vacc)+' ************')

			train_acc_list.append(avg_acc)
			val_acc_list.append(avg_vacc)

			# Track best performance, and save the model's state
			if avg_vloss < best_vloss:
				epoch_patience_cntr = 0
				best_vloss = avg_vloss
				model_path = os.path.join(save_path, 'weights-'+filename_suffix+'.pt')
				torch.save(model.state_dict(), model_path)
			else:
				print('*** validation loss did not improve ***'+str(epoch_patience_cntr))
				epoch_patience_cntr += 1

			torch.cuda.empty_cache()

			# save the model at the end of the epoch any way
			model_path = os.path.join(save_path, 'weights-'+filename_suffix+'.pt')
			torch.save(model.state_dict(), model_path)

			# adaptive learning rate
			# if validation loss does not improve for a number of epochs, decrease the learning rate
			if epoch_patience_cntr == 7 and learning_rate > min_learning_rate:    # halve the learning rate
				print(' -------------- reducing learning rate')
				learning_rate = learning_rate/5
				print('new learning_rate = '+str(learning_rate))
				epoch_patience_cntr = 0
				# reconfigure the optimizer:
				optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

			if early_stopping and epoch_patience_cntr == patience:
				# stop training
				print('********** stopping training **********')
				model_path = os.path.join(save_path, 'weights-'+filename_suffix+'.pt')
				torch.save(model.state_dict(), model_path)
				break

	val_loss_list = list(map(lambda x: float(x), val_loss_list))
	val_loss1_list = list(map(lambda x: float(x), val_loss1_list))
	val_loss2_list = list(map(lambda x: float(x), val_loss2_list))
	#print(val_loss_list)


	save_dict = {'learning_rate':learning_rate_list, 'val_loss':val_loss_list, 'train_loss':train_loss_list, 'train_acc':train_acc_list, 'val_acc':val_acc_list,
'train_triplet_loss':train_loss1_list, 'train_crossentropy_loss':train_loss2_list, 'val_triplet_loss':val_loss1_list, 'val_crossentropy_loss':val_loss2_list }
	save_loss_path = os.path.join(save_path,'loss-'+filename_suffix+'.pkl')
	with open (save_loss_path, 'wb') as handle:
		pickle.dump(save_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)
