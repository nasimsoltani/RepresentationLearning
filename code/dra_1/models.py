import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_seq_len=1024, d_model=256, nhead=4, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(Decoder, self).__init__()
        
        # Batchnorm for the input latent vector
        # self.input_bn = nn.BatchNorm1d(latent_dim)

        # --- RF Decoder ---
        # A CNN-based decoder is better suited for signal generation
        self.rf_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (32, 16)),  # (batch, 32, 16)
            nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1),   # Out: 64x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1),  # Out: 128x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # Out: 64x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # Out: 32x256
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),   # Out: 16x512
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 2, kernel_size=4, stride=2, padding=1)     # Out: 2x1024
        )

        # --- CFO/Channel Decoder (Simpler: using ConvTranspose) ---
        # These signals are more structured, so a CNN-based decoder might still be effective.
        # It takes the flattened latent vector and upsamples it.
        cfo_channel_latent_flat_dim = latent_dim # Latent dim is already flat
        
        self.cfo_channel_decoder = nn.Sequential(
            nn.Linear(cfo_channel_latent_flat_dim, 256),
            nn.Unflatten(1, (16, 16)),
            nn.ConvTranspose1d(16, 32, kernel_size=4, stride=2, padding=1),  # Out: 32x32
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # Out: 16x64
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 2, kernel_size=5, stride=2, padding=1),   # Out: 2x125
            nn.Upsample(size=160, mode='linear', align_corners=False)        # Out: 2x160
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # z is expected to be (batch, 2, 256)
        
        # Flatten the latent vector for the linear layer
        z_flat = z.view(z.size(0), -1)
        
        # Normalize the entire input latent vector
        # z_norm = self.input_bn(z_flat)

        # --- RF Reconstruction ---
        rf_output = self.rf_decoder(z_flat)

        # --- CFO/Channel Reconstruction ---
        cfo_channel_output = self.cfo_channel_decoder(z_flat)
        cfo_output = cfo_channel_output
        channel_output = cfo_channel_output # They share the same reconstruction path

        return {
            'rf': rf_output,
            'cfo': cfo_output,
            'channel': channel_output
        }


# Example usage:
if __name__ == '__main__':
    latent_dim = 512  # As per the encoder output
    decoder = Decoder(latent_dim=latent_dim)
    
    # Dummy input
    z = torch.randn(32, 2, 256) # (batch, channels, length)
    
    reconstructions = decoder(z)
    
    print("Output shapes:")
    for task, data in reconstructions.items():
        print(f"  {task}: {data.shape}")
    
    # Expected output:
    #   rf: torch.Size([32, 2, 1024])
    #   cfo: torch.Size([32, 2, 160])
    #   channel: torch.Size([32, 2, 160]) 