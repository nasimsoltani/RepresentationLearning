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

class ComplexSequenceProjector(nn.Module):
	def __init__(self, input_seq_len, output_seq_len, hidden_dim=512):
		"""
		Projects real and imaginary parts of a sequence to a new length using separate MLPs,
		then concatenates them.
		
		Args:
			input_seq_len (int): The length of the input sequence (L).
			output_seq_len (int): The desired length of the output sequence (L').
			hidden_dim (int): The hidden dimension of the MLP.
		
		Shape:
			- Input: (B, 2, input_seq_len)
			- Output: (B, 2, output_seq_len)
		"""
		super().__init__()
		self.real_projector = nn.Sequential(
			nn.Linear(input_seq_len, hidden_dim),
			nn.LeakyReLU(negative_slope=0.01),
			nn.Linear(hidden_dim, output_seq_len)
		)
		self.imag_projector = nn.Sequential(
			nn.Linear(input_seq_len, hidden_dim),
			nn.LeakyReLU(negative_slope=0.01),
			nn.Linear(hidden_dim, output_seq_len)
		)

	def forward(self, x):
		# x shape: (B, 2, L)
		x_real = x[:, 0, :] # Shape: (B, L)
		x_imag = x[:, 1, :] # Shape: (B, L)

		proj_real = self.real_projector(x_real).unsqueeze(1) # (B, L') -> (B, 1, L')
		proj_imag = self.imag_projector(x_imag).unsqueeze(1) # (B, L') -> (B, 1, L')

		return torch.cat([proj_real, proj_imag], dim=1) # (B, 2, L')


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
			nn.LeakyReLU(negative_slope=0.01),                         # (B, out_dim) -> (B, out_dim)
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


class SEBlock(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SEBlock, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1)
		return x * y.expand_as(x)


class Encoder(nn.Module):
	def __init__(self, slice_size, output_dim=128, dropout=0.25):
		"""
		An encoder that transforms input through a sequence of convolutional blocks.
		Based on RFFingerprintingNet architecture, with added SE blocks.
		
		Args:
			slice_size (int): Input sequence length
			output_dim (int): Final output dimension for each of the 2 channels.
			dropout (float): Dropout probability for all layers
		
		Shape:
			- Input: (batch_size, 2, slice_size)
			- Output: (batch_size, 2, output_dim)
		
		Example:
			>>> encoder = Encoder(1024, 128)
			>>> x = torch.randn(32, 2, 1024)  # (batch_size=32, channels=2, seq_len=1024)
			>>> out = encoder(x)              # (batch_size=32, 2, output_dim=128)
		"""
		super(Encoder, self).__init__()
		self.output_dim = output_dim
		
		channel = 64
		# Convolutional layers
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
		
		self.bn0 = nn.BatchNorm1d(channel)
		self.bn1 = nn.BatchNorm1d(channel)
		self.bn2 = nn.BatchNorm1d(channel)
		self.bn3 = nn.BatchNorm1d(channel)
		self.bn4 = nn.BatchNorm1d(channel)
		self.bn5 = nn.BatchNorm1d(channel)
		self.bn6 = nn.BatchNorm1d(channel)
		self.bn7 = nn.BatchNorm1d(channel)
		self.bn8 = nn.BatchNorm1d(channel)
		self.bn9 = nn.BatchNorm1d(channel)

		self.pool1 = nn.MaxPool1d(2,2)
		self.flatten = nn.Flatten()
		self.relu = nn.LeakyReLU(negative_slope=0.01)
		
		# SE Blocks
		# self.se1 = SEBlock(channel)
		# self.se2 = SEBlock(channel)
		# self.se3 = SEBlock(channel)
		# self.se4 = SEBlock(channel)
		# self.se5 = SEBlock(channel)
		
		# Calculate the size after all pooling operations
		conv_output_size = channel * (slice_size // (2**5))
		
		# Final MLP layers
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(conv_output_size, 256),
			nn.LayerNorm(256),
			nn.LeakyReLU(negative_slope=0.01),
			nn.Dropout(dropout),
			nn.Linear(256, 2 * output_dim)
		)
		
		# Initialize weights using Kaiming initialization
		self.apply(self._init_weights)
	
	def _init_weights(self, module):
		"""Initialize weights for linear layers using Kaiming initialization"""
		if isinstance(module, nn.Linear):
			nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
			if module.bias is not None:
				nn.init.zeros_(module.bias)
	
	def forward(self, x):
		"""
		Forward pass through the encoder.
		
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, 2, slice_size)
		
		Returns:
			torch.Tensor: Output tensor of shape (batch_size, 2, output_dim)
		
		Shape:
			- Input: (B, 2, L)
			- Conv blocks: (B, 64, L) -> (B, 64, L/2) -> (B, 64, L/4) -> (B, 64, L/8) -> (B, 64, L/16) -> (B, 64, L/32)
			- Output: (B, 2, output_dim)
		"""
		# x shape: (B, 2, L)
		# Block 1
		x = self.relu(self.bn0(self.conv0(x)))  # shape: (B, 64, L)
		x = self.relu(self.bn1(self.conv1(x)))  # shape: (B, 64, L)
		# x = self.se1(x)
		x = self.pool1(x)  # shape: (B, 64, L/2)
		
		# Block 2 with residual
		residual = x
		x = self.relu(self.bn2(self.conv2(x)))  # shape: (B, 64, L/2)
		x = self.bn3(self.conv3(x))
		#x = self.se2(x)
		x = self.relu(x + residual)
		x = self.pool1(x)  # shape: (B, 64, L/4)
		
		# Block 3 with residual
		residual = x
		x = self.relu(self.bn4(self.conv4(x)))  # shape: (B, 64, L/4)
		x = self.bn5(self.conv5(x))
		#x = self.se3(x)
		x = self.relu(x + residual)
		x = self.pool1(x)  # shape: (B, 64, L/8)
		
		# Block 4 with residual
		residual = x
		x = self.relu(self.bn6(self.conv6(x)))  # shape: (B, 64, L/8)
		x = self.bn7(self.conv7(x))
		#x = self.se4(x)
		x = self.relu(x + residual)
		x = self.pool1(x)  # shape: (B, 64, L/16)
		
		# Block 5 with residual
		residual = x
		x = self.relu(self.bn8(self.conv8(x)))  # shape: (B, 64, L/16)
		x = self.bn9(self.conv9(x))
		#x = self.se5(x)
		x = self.relu(x + residual)
		x = self.pool1(x)  # shape: (B, 64, L/32)
		
		features = self.flatten(x)  # shape: (B, 64 * L/32)
		
		output = self.classifier(features)  # shape: (B, 2 * output_dim)
		return output.view(output.size(0), 2, self.output_dim)


class RFClassificationHead(nn.Module):
	def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.2):
		"""
		Classification head for RF fingerprinting task.
		
		Args:
			input_dim (int): Input dimension from encoder (2 * d2)
			num_classes (int): Number of classes to classify
			hidden_dim (int): Hidden dimension for intermediate layer
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, 2, d2)
			- Output: (batch_size, num_classes)
		
		Example:
			>>> head = RFClassificationHead(2*128, num_classes=16)
			>>> x = torch.randn(32, 2, 128)      # (B, 2, d2)
			>>> out = head(x)                 # (B, 16)
		"""
		super(RFClassificationHead, self).__init__()
		
		self.classifier = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),    # (B, input_dim) -> (B, hidden_dim)
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(negative_slope=0.01),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, num_classes)    # (B, hidden_dim) -> (B, num_classes)
		)

		# Add proper initialization
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
					
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, 2, d2)
		Returns:
			torch.Tensor: Logits of shape (batch_size, num_classes)
		"""
		x = x.view(x.size(0), -1)
		return self.classifier(x)
	

	


class ChannelEstimationHead(nn.Module):
	def __init__(self, input_dim, hidden_dim=256, output_length=52, dropout=0.2):
		"""
		Regression head for channel estimation task.
		
		Args:
			input_dim (int): Input dimension from encoder (2 * d2)
			hidden_dim (int): Hidden dimension for intermediate layer
			output_length (int): Length of output sequence (default: 52)
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, 2, d2)
			- Output: (batch_size, 2, output_length)
		
		Example:
			>>> head = ChannelEstimationHead(2*128)
			>>> x = torch.randn(32, 2, 128)      # (B, 2, d2)
			>>> out = head(x)                 # (B, 2, 52)
		"""
		super(ChannelEstimationHead, self).__init__()
		
		self.output_length = output_length
		
		# Shared features
		self.shared = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),    # (B, input_dim) -> (B, hidden_dim)
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(negative_slope=0.01),
			nn.Dropout(dropout)
		)
		
		# Separate heads for real and imaginary parts
		self.real_head = nn.Linear(hidden_dim, output_length)    # (B, hidden_dim) -> (B, output_length)
		self.imag_head = nn.Linear(hidden_dim, output_length)    # (B, hidden_dim) -> (B, output_length)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, 2, d2)
		Returns:
			torch.Tensor: Channel estimates of shape (batch_size, 2, output_length)
		"""
		x = x.view(x.size(0), -1)
		features = self.shared(x)                # (B, 2*d2) -> (B, hidden_dim)
		
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
			input_dim (int): Input dimension from encoder (2 * d2)
			hidden_dim (int): Hidden dimension for intermediate layer
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, 2, d2)
			- Output: (batch_size, 1)
		
		Example:
			>>> head = CFOEstimationHead(2*128)
			>>> x = torch.randn(32, 2, 128)      # (B, 2, d2)
			>>> out = head(x)                 # (B, 1)
		"""
		super(CFOEstimationHead, self).__init__()
		
		self.regressor = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),    # (B, input_dim) -> (B, hidden_dim)
			#nn.LayerNorm(hidden_dim),
			#nn.ELU(),
			nn.LeakyReLU(negative_slope=0.01),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim//2), # (B, hidden_dim) -> (B, hidden_dim//2)
			#nn.LayerNorm(hidden_dim//2),
			#nn.ELU(),
			nn.LeakyReLU(negative_slope=0.01),
		
			nn.Dropout(dropout),
			nn.Linear(hidden_dim//2, 1),          # (B, hidden_dim//2) -> (B, 1)
			#nn.Tanh()                             # Bound output to [-1, 1]
			#output is unbounded
			
		)
		# Add proper initialization
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, 2, d2)
		Returns:
			torch.Tensor: CFO estimate of shape (batch_size, 1)
		"""
		x = x.view(x.size(0), -1)
		return self.regressor(x)                 # (B, 2*d2) -> (B, 1)




