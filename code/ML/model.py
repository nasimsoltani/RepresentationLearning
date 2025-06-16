import torch
import torch.nn as nn

import torch.nn as nn
import numpy as np


class RFFingerprintingNet(nn.Module):
	""" input shape needs to be (b, 2, 1024) """

	def __init__(self, slice_size, num_classes, mtl=False, common_rep_dim=512):
		super(RFFingerprintingNet, self).__init__()
		self.mtl = mtl
		dropProb = 0.25

		if not self.mtl:
			channel = 64
			self.conv0 = nn.Conv1d(2, channel, kernel_size=7, padding="same")
			self.conv1 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
			self.conv2 = nn.Conv1d(channel, channel, kernel_size=7, padding="same")
			self.conv3 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
			self.conv4 = nn.Conv1d(channel, channel, kernel_size=7, padding="same")
			self.conv5 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
			self.conv6 = nn.Conv1d(channel, channel, kernel_size=7, padding="same")
			self.conv7 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
			self.conv8 = nn.Conv1d(channel, channel, kernel_size=7, padding="same")
			self.conv9 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
			
			self.pool1 = nn.MaxPool1d(2,2)
			self.flatten = nn.Flatten()
			self.relu = nn.ReLU()
			classifier_input_size = channel * (slice_size // (2**5))
		else:
			classifier_input_size = common_rep_dim
		
		self.classifier = nn.Sequential(
			nn.Dropout(dropProb),
			nn.Linear(classifier_input_size, 256),
			nn.ReLU(),
			nn.Dropout(dropProb),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(dropProb),
			nn.Linear(128, num_classes)
		)

	def forward(self, x, common_rep=None):
		# x shape: (B, 2, L)
		if self.mtl:
			if common_rep is None:
				raise ValueError("common_rep is required for MTL mode")
			features = common_rep  # shape: (B, common_rep_dim)
		else:
			x = self.relu(self.conv0(x))  # shape: (B, 64, L)
			x = self.relu(self.conv1(x))  # shape: (B, 64, L)
			x = self.pool1(x)  # shape: (B, 64, L/2)
		
			x = self.relu(self.conv2(x))  # shape: (B, 64, L/2)
			x = self.relu(self.conv3(x))  # shape: (B, 64, L/2)
			x = self.pool1(x)  # shape: (B, 64, L/4)
			
			x = self.relu(self.conv4(x))  # shape: (B, 64, L/4)
			x = self.relu(self.conv5(x))  # shape: (B, 64, L/4)
			x = self.pool1(x)  # shape: (B, 64, L/8)

			x = self.relu(self.conv6(x))  # shape: (B, 64, L/8)
			x = self.relu(self.conv7(x))  # shape: (B, 64, L/8)
			x = self.pool1(x)  # shape: (B, 64, L/16)
			
			x = self.relu(self.conv8(x))  # shape: (B, 64, L/16)
			x = self.relu(self.conv9(x))  # shape: (B, 64, L/16)
			x = self.pool1(x)  # shape: (B, 64, L/32)
		
			features = self.flatten(x)  # shape: (B, 64 * L/32)
		
		# classifier layers:
		# Dropout -> (B, 64 * L/32) or (B, common_rep_dim)
		# Linear -> (B, 256)
		# ReLU -> (B, 256)
		# Dropout -> (B, 256)
		# Linear -> (B, 128)
		# ReLU -> (B, 128)
		# Dropout -> (B, 128)
		# Linear -> (B, num_classes)
		return self.classifier(features)


if __name__ == '__main__':

	slice_size = 1024 
	num_classes = 16

	# create the model:
	model = RFFingerprintingNet(slice_size, num_classes)

	# print number of parameters in the model
	pp=0
	for p in list(model.parameters()):
		n=1
		for s in list(p.size()):
			n = n*s
		pp += n
	print('This model has ' +str(pp)+ ' parameters')

	input = torch.rand((1, 2, slice_size))
	
	output = model(input)

	print(output.shape)
