import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAdversary(nn.Module):
    """
    A simple MLP-based adversary that tries to reconstruct original RF data from latent representations.
    
    Args:
        latent_dim (int): Dimension of the latent representation (typically 2 * d2)
        output_shape (tuple): Shape of the output data to reconstruct (channels, seq_length)
        hidden_dim (int): Hidden dimension for the MLP layers
        dropout (float): Dropout probability
    
    Shape:
        - Input: (batch_size, latent_dim) - flattened latent representation
        - Output: (batch_size, channels, seq_length) - reconstructed data
    """
    def __init__(self, latent_dim, output_shape, hidden_dim=512, dropout=0.2):
        super(SimpleAdversary, self).__init__()
        
        self.output_shape = output_shape
        self.output_dim = output_shape[0] * output_shape[1]  # channels * seq_length
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
            # Removed Tanh to allow full data range reconstruction
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, latent_dim)
        Returns:
            torch.Tensor: Reconstructed data of shape (batch_size, channels, seq_length)
        """
        flat_output = self.net(z)  # (B, latent_dim) -> (B, channels * seq_length)
        return flat_output.view(-1, self.output_shape[0], self.output_shape[1])  # (B, channels, seq_length)


class ConvolutionalAdversary(nn.Module):
    """
    A more sophisticated adversary using transposed convolutions to reconstruct RF data.
    This follows an encoder-decoder architecture similar to autoencoders.
    
    Args:
        latent_dim (int): Dimension of the latent representation
        output_shape (tuple): Shape of the output data to reconstruct (channels, seq_length)
        hidden_dim (int): Hidden dimension for intermediate layers
        dropout (float): Dropout probability
    
    Shape:
        - Input: (batch_size, latent_dim)
        - Output: (batch_size, channels, seq_length)
    """
    def __init__(self, latent_dim, output_shape, hidden_dim=256, dropout=0.2):
        super(ConvolutionalAdversary, self).__init__()
        
        self.output_shape = output_shape
        channels, seq_length = output_shape
        
        # Calculate the size we need to start with for transposed convolutions
        # We'll use 5 transpose conv layers with stride 2, so we need seq_length / 32
        self.initial_length = max(seq_length // 32, 8)
        self.initial_channels = 128
        
        # MLP to expand latent to initial conv feature map
        self.fc_expand = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.initial_channels * self.initial_length),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # Transposed convolutional layers to upsample
        self.deconv_layers = nn.ModuleList([
            # Each layer doubles the sequence length
            nn.ConvTranspose1d(self.initial_channels, 64, kernel_size=4, stride=2, padding=1),  # 2x
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),  # 4x
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # 16x
            nn.ConvTranspose1d(16, channels, kernel_size=4, stride=2, padding=1),  # 32x
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(32),
            nn.BatchNorm1d(16),
        ])
        
        # Final layer to get exact output size
        self.final_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, latent_dim)
        Returns:
            torch.Tensor: Reconstructed data of shape (batch_size, channels, seq_length)
        """
        # Expand to initial feature map
        x = self.fc_expand(z)  # (B, latent_dim) -> (B, initial_channels * initial_length)
        x = x.view(-1, self.initial_channels, self.initial_length)  # (B, initial_channels, initial_length)
        
        # Apply transposed convolutions
        for i, deconv in enumerate(self.deconv_layers[:-1]):
            x = deconv(x)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final deconv layer without batch norm
        x = self.deconv_layers[-1](x)
        
        # Adjust to exact output size if needed
        target_length = self.output_shape[1]
        if x.size(2) != target_length:
            if x.size(2) > target_length:
                # Crop if too long
                x = x[:, :, :target_length]
            else:
                # Pad if too short
                pad_length = target_length - x.size(2)
                x = F.pad(x, (0, pad_length))
        
        # Final conv to ensure correct number of channels
        x = self.final_conv(x)
        # Removed tanh to allow full data range reconstruction
        
        return x


class ResidualAdversary(nn.Module):
    """
    An adversary with residual connections for better gradient flow.
    Combines MLP and convolutional approaches.
    
    Args:
        latent_dim (int): Dimension of the latent representation
        output_shape (tuple): Shape of the output data to reconstruct (channels, seq_length)
        hidden_dim (int): Hidden dimension for MLP layers
        dropout (float): Dropout probability
    """
    def __init__(self, latent_dim, output_shape, hidden_dim=512, dropout=0.2):
        super(ResidualAdversary, self).__init__()
        
        self.output_shape = output_shape
        channels, seq_length = output_shape
        
        # Initial MLP to expand latent representation
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
        )
        
        # Residual blocks in the MLP space
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, dropout) for _ in range(3)
        ])
        
        # Final projection to output
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, channels * seq_length)
            # Removed tanh to allow full data range reconstruction
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _make_residual_block(self, dim, dropout):
        """Create a residual block"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, latent_dim)
        Returns:
            torch.Tensor: Reconstructed data of shape (batch_size, channels, seq_length)
        """
        x = self.fc_layers(z)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
            x = F.leaky_relu(x, negative_slope=0.01)
        
        # Project to output
        x = self.output_projection(x)
        
        # Reshape to target shape
        return x.view(-1, self.output_shape[0], self.output_shape[1])


def get_adversary(adversary_type, latent_dim, output_shape, **kwargs):
    """
    Factory function to create adversary models.
    
    Args:
        adversary_type (str): Type of adversary ('simple', 'conv', 'residual')
        latent_dim (int): Dimension of the latent representation
        output_shape (tuple): Shape of the output data (channels, seq_length)
        **kwargs: Additional arguments for the adversary
    
    Returns:
        nn.Module: The adversary model
    """
    if adversary_type == 'simple':
        return SimpleAdversary(latent_dim, output_shape, **kwargs)
    elif adversary_type == 'conv':
        return ConvolutionalAdversary(latent_dim, output_shape, **kwargs)
    elif adversary_type == 'residual':
        return ResidualAdversary(latent_dim, output_shape, **kwargs)
    else:
        raise ValueError(f"Unknown adversary type: {adversary_type}") 