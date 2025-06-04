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


class IQNet(nn.Module):
	""" input shape needs to be (b, 2, 1024) """

	def __init__(self, slice_size, num_channels, num_classes):
		super(IQNet, self).__init__()
		dropProb = 0.25
		channel = 64  # original: 32
		self.conv1 = nn.Conv1d(num_channels, channel, kernel_size=7, padding="same")
		self.conv2 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
		self.pool1 = nn.MaxPool1d(2,2)
		self.pool2 = nn.MaxPool1d(3, 2, padding=1)
		self.flatten = nn.Flatten()
		self.hidden1 = nn.Linear(int(slice_size*8), 512) #original slice_size*0.5   
		self.hidden2 = nn.Linear(512, 256)
		self.out = nn.Linear(256, num_classes)
		self.drop = nn.Dropout(dropProb)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()

	def forward(self, x):
		x = self.relu(self.conv1(x))

		b = x
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv2(x))
		x = torch.add(x, b)
		x = self.pool1(x)
		c = x = self.drop(x)
		

		x = self.relu(self.conv2(x))
		x = self.relu(self.conv2(x))
		x = torch.add(x, c)
		x = self.pool2(x)
		d = x = self.drop(x)
	
		
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv2(x))
		x = torch.add(x, d)
		x = self.pool2(x)
		x = self.drop(x)
        
		x = self.flatten(x)
		x = self.relu(self.hidden1(x))
		x = self.drop(x)
		
		feature = x
		
		x = self.relu(self.hidden2(x))
		x = self.drop(x)
		x = self.out(x)  # no softmax: CrossEntropyLoss()
		return feature, x



class SpectrogramNet(nn.Module):
	""" input shape needs to be (b, 3, 90, 160) """

	def __init__(self, slice_size, num_channels, num_classes):
		super(SpectrogramNet, self).__init__()
		dropProb = 0.25
		channel = 32  # original: 32
		self.conv1 = nn.Conv2d(num_channels, channel, kernel_size=(7,7), padding="same")
		self.conv2 = nn.Conv2d(channel, channel, kernel_size=(5,5), padding="same")
		self.pool1 = nn.MaxPool2d((2,2))
		self.pool2 = nn.MaxPool2d((3,3), padding=1)
		self.flatten = nn.Flatten()
		self.hidden1 = nn.Linear(6528, 512) #original slice_size*0.5   
		self.hidden2 = nn.Linear(512, 256)
		self.out = nn.Linear(256, num_classes)
		self.drop = nn.Dropout(dropProb)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()

	def forward(self, x):
		x = self.relu(self.conv1(x))

		b = x
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv2(x))
		x = torch.add(x, b)
		x = self.pool1(x)
		c = x = self.drop(x)
		

		x = self.relu(self.conv2(x))
		x = self.relu(self.conv2(x))
		x = torch.add(x, c)
		x = self.pool2(x)
		d = x = self.drop(x)
	
		
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv2(x))
		x = torch.add(x, d)
		x = self.pool2(x)
		x = self.drop(x)
        
		x = self.flatten(x)
		x = self.relu(self.hidden1(x))
		x = self.drop(x)
		
		feature = x
		
		x = self.relu(self.hidden2(x))
		x = self.drop(x)
		x = self.out(x)  # no softmax: CrossEntropyLoss()
		return feature, x


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
