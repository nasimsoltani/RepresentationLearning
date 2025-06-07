import torch
import torch.nn as nn

import torch.nn as nn
import numpy as np


class RFFingerprintingNet(nn.Module):
	""" input shape needs to be (b, 2, 1024) """

	def __init__(self, slice_size, num_classes):
		
		super(RFFingerprintingNet, self).__init__()
		
		dropProb = 0.25
		channel = 64 
		self.conv0 = nn.Conv1d(2, channel, kernel_size=7, padding="same")
		self.conv1 = nn.Conv1d(channel, channel, kernel_size=7, padding="same")
		self.conv2 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
		self.pool1 = nn.MaxPool1d(2,2)
		self.flatten = nn.Flatten()
		self.hidden1 = nn.Linear(int(slice_size*2), 256) 
		self.hidden2 = nn.Linear(256, 128)
		self.out = nn.Linear(128, num_classes)
		self.drop = nn.Dropout(dropProb)
		self.relu = nn.ReLU()

	def forward(self, x):
		
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv2(x))
		x = self.pool1(x)

		# TODO: Ask if weight sharing is intentional.
		for _ in range(4):
			
			x = self.relu(self.conv1(x))
			x = self.relu(self.conv2(x))
			x = self.pool1(x)

		x = self.flatten(x)
		x = self.drop(x)
		x = self.relu(self.hidden1(x))
		x = self.drop(x)
		x = self.relu(self.hidden2(x))
		x = self.drop(x)
		x = self.out(x)
		
		return x