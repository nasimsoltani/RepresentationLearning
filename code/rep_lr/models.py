import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
Our Arch will have 3 components. 
1. Light projection layer, to project the data to a common dimension.
2. Common representation layer, to extract the common representation of the data
3. Classifier layer, to classify the data

The input data might come in different shapes. One shape is (B, 2, L), which is the RF data.
The other shape is (B, 2, 160), which is the CFO and channel data.

The light projection layer will project the data to a common dimension.
The common representation layer will extract the common representation of the data.
The classifier layer will classify the data.

"""

class LightProjectionLayer(nn.Module):
	def __init__(self, in_channels, out_channels, output_dim=512):
		"""
		Light projection layer that projects input features to a fixed output dimension using GAP.
		
		Args:
			in_channels (int): Number of input channels (e.g. 2 for both RF and CFO/Channel data)
			out_channels (int): Number of intermediate channels
			output_dim (int): Final output dimension after flattening and projection
		"""
		super(LightProjectionLayer, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.output_dim = output_dim
		
		# 1x1 conv to project channels
		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # (B, in_ch, L) -> (B, out_ch, L)
		self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)  # (B, out_ch, L) -> (B, out_ch, L)
		
		# Global average pooling to remove sequence dimension
		self.gap = nn.AdaptiveAvgPool1d(1)  # (B, out_ch, L) -> (B, out_ch, 1)
		
		# Final projection to desired output dimension
		self.projection = nn.Sequential(
			nn.Flatten(),  # (B, out_ch, 1) -> (B, out_ch)
			nn.Linear(out_channels, output_dim),  # (B, out_ch) -> (B, output_dim)
			nn.LayerNorm(output_dim),
			nn.ReLU()
		)

	def forward(self, x):
		"""
		Forward pass of the projection layer.
		
		Args:
			x (torch.Tensor): Input tensor of shape (B, in_channels, L) 
							 where L can be 1024 for RF or 160 for CFO/Channel
		
		Returns:
			torch.Tensor: Output tensor of shape (B, output_dim)
		"""
		x = self.conv1(x)  # (B, in_ch, L) -> (B, out_ch, L)
		x = self.conv2(x)  # (B, out_ch, L) -> (B, out_ch, L)
		x = self.gap(x)    # (B, out_ch, L) -> (B, out_ch, 1)
		x = self.projection(x)  # (B, out_ch, 1) -> (B, output_dim)
		return x


class ProjectionLayerMLP(nn.Module):
	def __init__(self, in_channels, out_channels, seq_length, output_dim=512):
		"""
		Projection layer that preserves sequence information before final projection.
		
		Args:
			in_channels (int): Number of input channels (e.g. 2 for both RF and CFO/Channel data)
			out_channels (int): Number of channels after conv projection
			seq_length (int): Length of input sequence (1024 for RF or 160 for CFO/Channel)
			output_dim (int): Final output dimension after MLP projection
		"""
		super(ProjectionLayerMLP, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.seq_length = seq_length
		self.output_dim = output_dim
		
		# Light conv projection to mix channels
		self.conv1 = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=1),  # (B, in_ch, L) -> (B, out_ch, L)
			nn.BatchNorm1d(out_channels),
			nn.ReLU()
		)
		
		# MLP projection after flattening
		flattened_dim = out_channels * seq_length
		self.mlp = nn.Sequential(
			nn.Flatten(),  # (B, out_ch, L) -> (B, out_ch * L)
			nn.Linear(flattened_dim, flattened_dim // 2),  # First reduce dimension
			nn.LayerNorm(flattened_dim // 2),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(flattened_dim // 2, output_dim),  # Project to final dimension
			nn.LayerNorm(output_dim),
			nn.ReLU()
		)

	def forward(self, x):
		"""
		Forward pass of the projection layer.
		
		Args:
			x (torch.Tensor): Input tensor of shape (B, in_channels, L)
							 where L is seq_length
		
		Returns:
			torch.Tensor: Output tensor of shape (B, output_dim)
		"""
		x = self.conv1(x)  # (B, in_ch, L) -> (B, out_ch, L)
		x = self.mlp(x)    # (B, out_ch, L) -> (B, output_dim)
		return x


class MLPBlock(nn.Module):
	def __init__(self, in_dim, out_dim, dropout=0.1):
		"""
		A basic MLP block with normalization, activation, and residual connection if dimensions match.
		
		Args:
			in_dim (int): Input dimension
			out_dim (int): Output dimension
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, in_dim)
			- Output: (batch_size, out_dim)
		
		Example:
			>>> block = MLPBlock(256, 128)
			>>> x = torch.randn(32, 256)    # (batch_size=32, in_dim=256)
			>>> out = block(x)              # (batch_size=32, out_dim=128)
		"""
		super(MLPBlock, self).__init__()
		self.has_residual = (in_dim == out_dim)
		
		# Transform: (B, in_dim) -> (B, out_dim)
		self.mlp = nn.Sequential(
			nn.Linear(in_dim, out_dim),        # (B, in_dim) -> (B, out_dim)
			nn.LayerNorm(out_dim),             # (B, out_dim) -> (B, out_dim)
			nn.ReLU(),                         # (B, out_dim) -> (B, out_dim)
			nn.Dropout(dropout)                 # (B, out_dim) -> (B, out_dim)
		)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, in_dim)
		Returns:
			torch.Tensor: Output tensor of shape (batch_size, out_dim)
		"""
		out = self.mlp(x)                      # (B, in_dim) -> (B, out_dim)
		if self.has_residual:
			out = out + x                      # (B, out_dim) + (B, out_dim) -> (B, out_dim)
		return out


class Encoder(nn.Module):
	def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, dropout=0.1):
		"""
		An encoder that transforms input through a sequence of MLP blocks.
		
		Args:
			input_dim (int): Input dimension
			hidden_dims (list): List of hidden dimensions for intermediate layers
			output_dim (int): Final output dimension
			dropout (float): Dropout probability for all layers
		
		Shape:
			- Input: (batch_size, input_dim)
			- Output: (batch_size, output_dim)
		
		Example for a 3-layer encoder:
			>>> encoder = Encoder(1024, [512, 256], 128)
			>>> x = torch.randn(32, 1024)     # (batch_size=32, input_dim=1024)
			>>> out = encoder(x)              # (batch_size=32, output_dim=128)
			
			The data will flow through these dimensions:
			(B, 1024) -> (B, 512) -> (B, 256) -> (B, 128)
		"""
		super(Encoder, self).__init__()
		
		# Build the layer dimensions including input and output
		layer_dims = [input_dim] + hidden_dims + [output_dim]
		
		# Create sequential MLP blocks
		layers = []
		for i in range(len(layer_dims) - 1):
			# Each block: (B, layer_dims[i]) -> (B, layer_dims[i+1])
			layers.append(MLPBlock(layer_dims[i], layer_dims[i+1], dropout))
		
		self.encoder = nn.Sequential(*layers)
		
		# Initialize weights using Kaiming initialization
		self.apply(self._init_weights)
	
	def _init_weights(self, module):
		"""Initialize weights for linear layers using Kaiming initialization"""
		if isinstance(module, nn.Linear):
			nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
			if module.bias is not None:
				nn.init.zeros_(module.bias)
	
	def forward(self, x):
		"""
		Forward pass through the encoder.
		
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
		
		Returns:
			torch.Tensor: Output tensor of shape (batch_size, output_dim)
		
		Shape:
			- Input: (B, input_dim)
			- Hidden: (B, hidden_dims[0]) -> (B, hidden_dims[1]) -> ...
			- Output: (B, output_dim)
		"""
		return self.encoder(x)  # (B, input_dim) -> (B, output_dim)


class RFClassificationHead(nn.Module):
	def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.2):
		"""
		Classification head for RF fingerprinting task.
		
		Args:
			input_dim (int): Input dimension from encoder
			num_classes (int): Number of classes to classify
			hidden_dim (int): Hidden dimension for intermediate layer
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, input_dim)
			- Output: (batch_size, num_classes)
		
		Example:
			>>> head = RFClassificationHead(512, num_classes=16)
			>>> x = torch.randn(32, 512)      # (B, input_dim)
			>>> out = head(x)                 # (B, 16)
		"""
		super(RFClassificationHead, self).__init__()
		
		self.classifier = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),    # (B, input_dim) -> (B, hidden_dim)
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, num_classes)    # (B, hidden_dim) -> (B, num_classes)
		)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
		Returns:
			torch.Tensor: Logits of shape (batch_size, num_classes)
		"""
		return self.classifier(x)


class ChannelEstimationHead(nn.Module):
	def __init__(self, input_dim, hidden_dim=256, output_length=52, dropout=0.2):
		"""
		Regression head for channel estimation task.
		
		Args:
			input_dim (int): Input dimension from encoder
			hidden_dim (int): Hidden dimension for intermediate layer
			output_length (int): Length of output sequence (default: 52)
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, input_dim)
			- Output: (batch_size, 2, output_length)
		
		Example:
			>>> head = ChannelEstimationHead(512)
			>>> x = torch.randn(32, 512)      # (B, input_dim)
			>>> out = head(x)                 # (B, 2, 52)
		"""
		super(ChannelEstimationHead, self).__init__()
		
		self.output_length = output_length
		
		# Shared features
		self.shared = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),    # (B, input_dim) -> (B, hidden_dim)
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout)
		)
		
		# Separate heads for real and imaginary parts
		self.real_head = nn.Linear(hidden_dim, output_length)    # (B, hidden_dim) -> (B, output_length)
		self.imag_head = nn.Linear(hidden_dim, output_length)    # (B, hidden_dim) -> (B, output_length)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
		Returns:
			torch.Tensor: Channel estimates of shape (batch_size, 2, output_length)
		"""
		features = self.shared(x)                # (B, input_dim) -> (B, hidden_dim)
		
		# Get real and imaginary components
		real = self.real_head(features)          # (B, hidden_dim) -> (B, output_length)
		imag = self.imag_head(features)          # (B, hidden_dim) -> (B, output_length)
		
		# Stack real and imaginary parts
		out = torch.stack([real, imag], dim=1)   # [(B, output_length), (B, output_length)] -> (B, 2, output_length)
		return out


class CFOEstimationHead(nn.Module):
	def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
		"""
		Regression head for CFO estimation task.
		
		Args:
			input_dim (int): Input dimension from encoder
			hidden_dim (int): Hidden dimension for intermediate layer
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, input_dim)
			- Output: (batch_size, 1)
		
		Example:
			>>> head = CFOEstimationHead(512)
			>>> x = torch.randn(32, 512)      # (B, input_dim)
			>>> out = head(x)                 # (B, 1)
		"""
		super(CFOEstimationHead, self).__init__()
		
		self.regressor = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),    # (B, input_dim) -> (B, hidden_dim)
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim//2), # (B, hidden_dim) -> (B, hidden_dim//2)
			nn.LayerNorm(hidden_dim//2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim//2, 1),          # (B, hidden_dim//2) -> (B, 1)
			nn.Tanh()                             # Bound output to [-1, 1]
		)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
		Returns:
			torch.Tensor: CFO estimate of shape (batch_size, 1)
		"""
		return self.regressor(x)                 # (B, input_dim) -> (B, 1)




