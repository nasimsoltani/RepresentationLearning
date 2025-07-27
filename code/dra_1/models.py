import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, dropout=0.1):
        super(Decoder, self).__init__()
        
        # Shared decoder backbone - processes the latent vector into intermediate features
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Unflatten(1, (64, 16)),  # (batch, 64, 16)
        )
        
        # Shared convolutional layers for upsampling
        self.shared_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1),   # Out: 128x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # Out: 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),    # Out: 32x128
            nn.ReLU(inplace=True),
        )
        
        # Task-specific output heads
        # RF head - minimal, separate MLP decoder from latent vector
        self.rf_head = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2*1024),
            nn.Unflatten(1, (2, 1024)),
        )
        
        # CFO head - needs to go from 128 to 160 samples
        self.cfo_head = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),    # Out: 16x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=1, padding=1),     # Out: 8x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(8, 2, kernel_size=3, stride=1, padding=0),      # Out: 2x130
            nn.Upsample(size=160, mode='linear', align_corners=False)          # Out: 2x160
        )
        
        # Channel head - same as CFO since they have same output size
        self.channel_head = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),    # Out: 16x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=1, padding=1),     # Out: 8x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(8, 2, kernel_size=3, stride=1, padding=0),      # Out: 2x130
            nn.Upsample(size=160, mode='linear', align_corners=False)          # Out: 2x160
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for RF head to ensure proper scale
        for m in self.rf_head.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # Use Xavier initialization with higher gain for RF head
                nn.init.xavier_normal_(m.weight, gain=2.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z, task):
        # z is expected to be (batch, 2, 256) -> flatten to (batch, 512)
        z_flat = z.view(z.size(0), -1)
        
        if task == 'rf':
            rf_output = self.rf_head(z_flat)
            return rf_output
        else:
            # Shared processing
            shared_features = self.shared_decoder(z_flat)  # (batch, 64, 16)
            conv_features = self.shared_conv(shared_features)  # (batch, 32, 128)
            if task == 'cfo':
                return self.cfo_head(conv_features)
            elif task == 'channel':
                return self.channel_head(conv_features)
            else:
                raise ValueError(f"Unknown task: {task}")


# Example usage:
if __name__ == '__main__':
    latent_dim = 512  # As per the encoder output
    decoder = Decoder(latent_dim=latent_dim)
    
    # Dummy input
    z = torch.randn(32, 2, 256) # (batch, channels, length)
    
    reconstructions = decoder(z, 'rf')
    
    print("Output shapes:")
    for task, data in reconstructions.items():
        print(f"  {task}: {data.shape}")
    
    # Expected output:
    #   rf: torch.Size([32, 2, 1024])
    #   cfo: torch.Size([32, 2, 160])
    #   channel: torch.Size([32, 2, 160]) 