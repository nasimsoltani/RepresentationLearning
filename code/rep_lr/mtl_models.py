import torch
import torch.nn as nn

class EarlyFusionNet(nn.Module):
    def __init__(self, common_rep_dim=512):
        super().__init__()
        # Projection for RF_X (2, 1024)
        self.rf_projection = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3), # B, 64, 512
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # B, 64, 256
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # B, 128, 128
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # B, 128, 64
            nn.Flatten(),
            nn.Linear(128 * 64, 256),
            nn.ReLU()
        )

        # Projection for CFO_X and Channel_X (2, 160)
        self.other_projection = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3), # B, 64, 80
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # B, 64, 40
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2), # B, 128, 40
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # B, 128, 20
            nn.Flatten(),
            nn.Linear(128 * 20, 256),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(256 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, common_rep_dim),
            nn.ReLU()
        )

    def forward(self, rf_x, cfo_x, channel_x):
        rf_feat = self.rf_projection(rf_x)
        cfo_feat = self.other_projection(cfo_x)
        channel_feat = self.other_projection(channel_x)

        fused_features = torch.cat((rf_feat, cfo_feat, channel_feat), dim=1)
        common_representation = self.mlp(fused_features)
        return common_representation

class RFFingerprintingNetMTL(nn.Module):
    def __init__(self, slice_size, num_classes, common_rep_dim=512):
        super().__init__()
        dropProb = 0.25
        channel = 64 
        self.conv0 = nn.Conv1d(2, channel, kernel_size=7, padding="same")
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
        self.pool1 = nn.MaxPool1d(2,2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        self.feature_extractor = nn.Sequential(
            self.relu,
            nn.Conv1d(2, channel, kernel_size=7, padding="same"),
            self.relu,
            nn.Conv1d(channel, channel, kernel_size=5, padding="same"),
            self.relu,
            nn.MaxPool1d(2,2)
        )
        
        # Intermediate feature dimension after conv layers and pooling
        # This needs to be calculated based on the convolutions.
        # After one pooling of 2: 1024 -> 512. 4 more: 512 -> 256 -> 128 -> 64 -> 32
        # final feature map size is (channel, 32)
        self.flattened_size = channel * (slice_size // (2**5)) 
        
        self.fusion = nn.Linear(self.flattened_size + common_rep_dim, int(slice_size*2))

        self.classifier = nn.Sequential(
            nn.Linear(int(slice_size*2), 256),
            nn.ReLU(),
            nn.Dropout(dropProb),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropProb),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, common_rep=None):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)

        for _ in range(4):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool1(x)

        x = self.flatten(x)
        
        if common_rep is not None:
            x = torch.cat((x, common_rep), dim=1)
            x = self.fusion(x)
            
        x = self.classifier(x)
        return x

class ChannelNetMTL(nn.Module):
    def __init__(self, common_rep_dim=512):
        super().__init__()
        self.mlp_real = nn.Sequential(
            nn.Linear(160, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mlp_imag = nn.Sequential(
            nn.Linear(160, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.fusion_real = nn.Linear(256 + common_rep_dim, 256)
        self.fusion_imag = nn.Linear(256 + common_rep_dim, 256)

        self.out_real = nn.Linear(256, 52)
        self.out_imag = nn.Linear(256, 52)

    def forward(self, x, common_rep=None):
        real_in = x[:, 0, :]
        imag_in = x[:, 1, :]

        real_feat = self.mlp_real(real_in)
        imag_feat = self.mlp_imag(imag_in)
        
        if common_rep is not None:
            real_feat = self.fusion_real(torch.cat((real_feat, common_rep), dim=1))
            imag_feat = self.fusion_imag(torch.cat((imag_feat, common_rep), dim=1))

        real_out = self.out_real(real_feat)
        imag_out = self.out_imag(imag_feat)
        
        output = torch.stack((real_out, imag_out), dim=1)
        return output

class CFONetLargeMTL(nn.Module):
    def __init__(self, output_dim=1, common_rep_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.fusion = nn.Linear(128 * 14 + common_rep_dim, 128 * 14)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 14, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, common_rep=None):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        
        if common_rep is not None:
            x = self.fusion(torch.cat((x, common_rep), dim=1))

        x = self.classifier(x)
        return x.unsqueeze(-1)

class CFONetSmallMTL(nn.Module):
    def __init__(self, output_dim=1, common_rep_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.fusion = nn.Linear(128 * 35 + common_rep_dim, 128 * 35)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 35, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, common_rep=None):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten

        if common_rep is not None:
            x = self.fusion(torch.cat((x, common_rep), dim=1))

        x = self.classifier(x)
        return x.unsqueeze(-1) 