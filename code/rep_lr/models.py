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

class UpsamplingProjector(nn.Module):
	def __init__(self, output_seq_len):
		"""
		Projects a sequence to a new length using linear interpolation,
		which preserves the sequence's structure.
		
		Args:
			output_seq_len (int): The desired length of the output sequence (L').
		
		Shape:
			- Input: (B, 2, input_seq_len)
			- Output: (B, 2, output_seq_len)
		"""
		super().__init__()
		self.output_seq_len = output_seq_len

	def forward(self, x):
		# x shape: (B, 2, L)
		return F.interpolate(x, size=self.output_seq_len, mode='linear', align_corners=False)


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
    def __init__(self, slice_size, output_dim=128, dropout=0.25, num_blocks=5, input_channels=2):
        """
        An encoder that transforms input through a sequence of convolutional blocks.
        Based on RFFingerprintingNet architecture, with added SE blocks.
        
        Args:
            slice_size (int): Input sequence length
            output_dim (int): Final output dimension for each of the 2 channels.
            dropout (float): Dropout probability for all layers
            num_blocks (int): Number of convolutional blocks (default: 5)
            input_channels (int): Number of input channels (default: 2)
        
        Shape:
            - Input: (batch_size, input_channels, slice_size)
            - Output: (batch_size, 2, output_dim)
        
        Example:
            >>> encoder = Encoder(1024, 128, num_blocks=3)  # Shallow network
            >>> encoder = Encoder(1024, 128, num_blocks=7)  # Deep network
            >>> x = torch.randn(32, 2, 1024)
            >>> out = encoder(x)              # (batch_size=32, 2, output_dim=128)
        """
        super(Encoder, self).__init__()
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        channel = 64

        # Dynamic convolutional and batchnorm layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.conv_layers.append(nn.Conv1d(input_channels, channel, kernel_size=7, padding="same"))
                self.bn_layers.append(nn.BatchNorm1d(channel))
                self.conv_layers.append(nn.Conv1d(channel, channel, kernel_size=5, padding="same"))
                self.bn_layers.append(nn.BatchNorm1d(channel))
            else:
                self.conv_layers.append(nn.Conv1d(channel, channel, kernel_size=7, padding="same"))
                self.bn_layers.append(nn.BatchNorm1d(channel))
                self.conv_layers.append(nn.Conv1d(channel, channel, kernel_size=5, padding="same"))
                self.bn_layers.append(nn.BatchNorm1d(channel))

        self.pool = nn.MaxPool1d(2,2)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU(negative_slope=0.1)

        # Calculate the size after all pooling operations
        conv_output_size = channel * (slice_size // (2 ** num_blocks))

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
        """
        # x shape: (B, 2, L)
        for i in range(self.num_blocks):
            idx = i * 2
            # First conv + BN + ReLU
            x = self.relu(self.bn_layers[idx](self.conv_layers[idx](x)))
            # Second conv + BN
            x = self.bn_layers[idx+1](self.conv_layers[idx+1](x))
            # Residual connection for all but the first block
            if i > 0:
                x = self.relu(x + residual)
            else:
                x = self.relu(x)
            residual = x
            x = self.pool(x)
        features = self.flatten(x)
        output = self.classifier(features)
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
		self.input_dim = input_dim
		
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
		self.input_dim = input_dim
		
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
		self.input_dim = input_dim
		
		self.regressor = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),    # (B, input_dim) -> (B, hidden_dim)
			nn.BatchNorm1d(hidden_dim),          # Use BatchNorm instead of LayerNorm for regression
			nn.ReLU(),                           # Use ReLU instead of LeakyReLU for regression
			nn.Dropout(dropout),
			
			nn.Linear(hidden_dim, hidden_dim//2), # (B, hidden_dim) -> (B, hidden_dim//2)
			nn.BatchNorm1d(hidden_dim//2),
			nn.ReLU(),
			nn.Dropout(dropout),
			
			nn.Linear(hidden_dim//2, hidden_dim//4), # Add one more layer for better capacity
			nn.BatchNorm1d(hidden_dim//4),
			nn.ReLU(),
			nn.Dropout(dropout//2),              # Reduce dropout for final layers
			
			nn.Linear(hidden_dim//4, 1)          # (B, hidden_dim//4) -> (B, 1)
			# No activation - let the model learn the full range
		)
		
		# Better initialization for regression
		self._init_weights()
	
	def _init_weights(self):
		"""Initialize weights specifically for regression task."""
		for m in self.modules():
			if isinstance(m, nn.Linear):
				# Use Xavier/Glorot initialization for regression
				nn.init.xavier_normal_(m.weight, gain=1.0)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
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


class SimpleCFOEstimationHead(nn.Module):
	def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
		"""
		Simple regression head for CFO estimation, based on successful baseline.
		
		Args:
			input_dim (int): Input dimension from encoder (2 * d2)
			hidden_dim (int): Hidden dimension for intermediate layer
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, 2, d2)
			- Output: (batch_size, 1)
		"""
		super(SimpleCFOEstimationHead, self).__init__()
		self.input_dim = input_dim
		
		# Simple 2-layer architecture like the successful baseline
		self.regressor = nn.Sequential(
			nn.Flatten(),                        # (B, 2, d2) -> (B, 2*d2) 
			nn.Linear(input_dim, hidden_dim),    # (B, 2*d2) -> (B, hidden_dim)
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 1)             # (B, hidden_dim) -> (B, 1)
		)
		
		# Simple initialization
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, 2, d2)
		Returns:
			torch.Tensor: CFO estimate of shape (batch_size, 1)
		"""
		return self.regressor(x)


class DirectCFOEstimationHead(nn.Module):
	def __init__(self, input_seq_len=256, hidden_dim=128, dropout=0.1):
		"""
		Direct CFO estimation that works on projected sequences without encoder.
		
		Args:
			input_seq_len (int): Length of projected sequence
			hidden_dim (int): Hidden dimension for processing
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, 2, input_seq_len)  # Projected CFO data
			- Output: (batch_size, 1)
		"""
		super(DirectCFOEstimationHead, self).__init__()
		
		# Process each channel separately then combine
		self.channel_processor = nn.Sequential(
			nn.Conv1d(2, 16, kernel_size=7, padding=3),  # Light processing
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(32),  # Reduce to fixed size
			nn.Flatten(),  # (B, 16*32) = (B, 512)
		)
		
		self.regressor = nn.Sequential(
			nn.Linear(512, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 64),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(64, 1)
		)
		
		# Initialize weights
		for m in self.modules():
			if isinstance(m, (nn.Linear, nn.Conv1d)):
				nn.init.xavier_normal_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Projected CFO data of shape (batch_size, 2, seq_len)
		Returns:
			torch.Tensor: CFO estimate of shape (batch_size, 1)
		"""
		features = self.channel_processor(x)  # (B, 2, L) -> (B, 512)
		return self.regressor(features)       # (B, 512) -> (B, 1)


class CFOAdaptiveHead(nn.Module):
	def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
		"""
		CFO head with pre-processing to adapt shared encoder features for CFO estimation.
		This works with MTL by adding CFO-specific processing after the shared encoder.
		
		Args:
			input_dim (int): Input dimension from shared encoder (2 * d2)
			hidden_dim (int): Hidden dimension for CFO-specific processing
			dropout (float): Dropout probability
		
		Shape:
			- Input: (batch_size, 2, d2) from shared encoder
			- Output: (batch_size, 1)
		"""
		super(CFOAdaptiveHead, self).__init__()
		self.input_dim = input_dim
		
		# CFO-specific feature adaptation
		self.cfo_adapter = nn.Sequential(
			# Reshape and process encoder features for CFO
			nn.Flatten(),  # (B, 2, d2) -> (B, 2*d2)
			nn.Linear(input_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),  # Normalize features for regression
			nn.ReLU(),
			nn.Dropout(dropout),
			
			# CFO-specific transformation
			nn.Linear(hidden_dim, hidden_dim//2),
			nn.LayerNorm(hidden_dim//2),
			nn.ReLU(),
			nn.Dropout(dropout//2),
			
			# Final regression layer
			nn.Linear(hidden_dim//2, 1)
		)
		
		# Better initialization for regression in MTL setting
		self._init_weights()
	
	def _init_weights(self):
		"""Initialize weights for stable MTL training."""
		for m in self.modules():
			if isinstance(m, nn.Linear):
				# Use smaller initialization for MTL stability
				nn.init.xavier_normal_(m.weight, gain=0.5)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.LayerNorm):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): Features from shared encoder of shape (batch_size, 2, d2)
		Returns:
			torch.Tensor: CFO estimate of shape (batch_size, 1)
		"""
		return self.cfo_adapter(x)


class TaskAdaptiveEncoder(nn.Module):
	def __init__(self, slice_size, output_dim=128, dropout=0.25, num_blocks=3, input_channels=2):
		"""
		Enhanced encoder that works better for multiple tasks including CFO.
		Less aggressive pooling and more gradual feature extraction.
		
		Args:
			slice_size (int): Input sequence length
			output_dim (int): Final output dimension for each of the 2 channels
			dropout (float): Dropout probability
			num_blocks (int): Number of convolutional blocks (fewer for CFO compatibility)
			input_channels (int): Number of input channels (default: 2)
		"""
		super(TaskAdaptiveEncoder, self).__init__()
		self.output_dim = output_dim
		self.num_blocks = num_blocks
		
		# Use smaller number of blocks and gentler pooling for CFO compatibility
		self.layers = nn.ModuleList()
		
		channels = [input_channels, 32, 64, 64]  # Gentler channel progression
		
		for i in range(num_blocks):
			in_ch = channels[i]
			out_ch = channels[i+1]
			
			block = nn.Sequential(
				nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
				nn.BatchNorm1d(out_ch),
				nn.ReLU(),
				nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
				nn.BatchNorm1d(out_ch),
				nn.ReLU(),
				# Gentler pooling - adaptive instead of fixed stride
				nn.AdaptiveMaxPool1d(slice_size // (2 ** (i+1))) if i < num_blocks-1 else nn.Identity()
			)
			self.layers.append(block)
		
		# Calculate final feature size based on actual num_blocks used
		final_length = slice_size // (2 ** (num_blocks-1))
		actual_output_channels = channels[num_blocks]  # Use the actual output channels for num_blocks
		conv_output_size = actual_output_channels * final_length
		
		# Final projection with residual-like connection
		self.feature_projection = nn.Sequential(
			nn.AdaptiveAvgPool1d(32),  # Fixed size output
			nn.Flatten(),
			nn.Linear(actual_output_channels * 32, 256),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(256, 2 * output_dim)
		)
		
		self.apply(self._init_weights)
	
	def _init_weights(self, module):
		"""Initialize weights for stable multi-task training."""
		if isinstance(module, nn.Linear):
			nn.init.xavier_normal_(module.weight, gain=0.8)  # Conservative gain for MTL
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Conv1d):
			nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
			if module.bias is not None:
				nn.init.zeros_(module.bias)
	
	def forward(self, x):
		"""
		Forward pass through task-adaptive encoder.
		Args:
			x (torch.Tensor): Input of shape (batch_size, 2, slice_size)
		Returns:
			torch.Tensor: Features of shape (batch_size, 2, output_dim)
		"""
		# Process through convolutional blocks
		for layer in self.layers:
			x = layer(x)
		
		# Final feature projection
		features = self.feature_projection(x)
		return features.view(features.size(0), 2, self.output_dim)




