import torch
import torch.nn as nn

import torch.nn as nn
import numpy as np



#Input Bx2xL, Output Bxnum_classes, Classification Task
class RFFingerprintingNet(nn.Module):
	""" input shape needs to be (b, 2, 1024) """

	def __init__(self, slice_size, num_classes, mtl=False, common_rep_dim=512):
		super(RFFingerprintingNet, self).__init__()
		self.mtl = mtl
		dropProb = 0.25

		if not self.mtl:
			channel = 64 
			self.conv0 = nn.Conv1d(2, channel, kernel_size=7, padding="same")
			self.conv1 = nn.Conv1d(channel, channel, kernel_size=7, padding="same")
			self.conv2 = nn.Conv1d(channel, channel, kernel_size=5, padding="same")
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
		if self.mtl:
			if common_rep is None:
				raise ValueError("common_rep is required for MTL mode")
			features = common_rep
		else:
			x = self.relu(self.conv0(x))
			x = self.relu(self.conv2(x))
			x = self.pool1(x)

			# TODO: Ask if weight sharing is intentional.
			for _ in range(4):
				
				x = self.relu(self.conv1(x))
				x = self.relu(self.conv2(x))
				x = self.pool1(x)

			features = self.flatten(x)
		return self.classifier(features)

# Input Bx2x160, Output Bx2x52, Regression Task
class ChannelNet(nn.Module):
    """
    Channel Estimation Network.
    Input shape: (B, 2, 160)
    Output shape: (B, 2, 52)
    """
    def __init__(self, mtl=False, common_rep_dim=512):
        super(ChannelNet, self).__init__()
        self.mtl = mtl

        if not self.mtl:
            # MLP for the real part of the CSI
            self.mlp_real = nn.Sequential(
                nn.Linear(160, 512),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 52)
            )

            # MLP for the imaginary part of the CSI
            self.mlp_imag = nn.Sequential(
                nn.Linear(160, 512),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 52)
            )
        else:
            self.mtl_mlp = nn.Sequential(
                nn.Linear(common_rep_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 104) # 2 * 52 for real and imag
            )

    def forward(self, x, common_rep=None):
        if self.mtl:
            if common_rep is None:
                raise ValueError("common_rep is required for MTL mode")
            output = self.mtl_mlp(common_rep)
            output = output.view(output.size(0), 2, 52)
        else:
            # x is expected to be of shape (B, 2, 160)
            real_in = x[:, 0, :]
            imag_in = x[:, 1, :]

            real_out = self.mlp_real(real_in)  # (B, 52)
            imag_out = self.mlp_imag(imag_in)  # (B, 52)
            
            # Stack the outputs to get (B, 2, 52)
            output = torch.stack((real_out, imag_out), dim=1)
        
        return output


#Input Bx2x160, Output Bx1x1, Regression Task
#TODO: Ask if we should use tanh or not. Tanh ouptut is in rnage -1,1
class CFONetLarge(nn.Module):
    def __init__(self, output_dim=1, mtl=False, common_rep_dim=512):
        super(CFONetLarge, self).__init__()
        self.mtl = mtl
        
        if not self.mtl:
            self.features = nn.Sequential(
                # Block 1
                nn.Conv1d(2, 64, kernel_size=3), nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                # Block 2
                nn.Conv1d(128, 64, kernel_size=3), nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                # Block 3
                nn.Conv1d(128, 64, kernel_size=3), nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            classifier_input_size = 128 * 14
        else:
            classifier_input_size = common_rep_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
            #nn.Tanh()
        )

    def forward(self, x, common_rep=None):
        if self.mtl:
            if common_rep is None:
                raise ValueError("common_rep is required for MTL mode")
            features = common_rep
        else:
            x = self.features(x)
            features = x.view(x.size(0), -1) # Flatten
        
        x = self.classifier(features)
        return x.unsqueeze(-1)


class CFONetSmall(nn.Module):
    def __init__(self, output_dim=1, mtl=False, common_rep_dim=512):
        super(CFONetSmall, self).__init__()
        self.mtl = mtl

        if not self.mtl:
            self.features = nn.Sequential(
                # Block 1
                nn.Conv1d(2, 64, kernel_size=3), nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                # Block 2
                nn.Conv1d(128, 64, kernel_size=3), nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            classifier_input_size = 128 * 35
        else:
            classifier_input_size = common_rep_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
            #nn.Tanh()
        )

    def forward(self, x, common_rep=None):
        if self.mtl:
            if common_rep is None:
                raise ValueError("common_rep is required for MTL mode")
            features = common_rep
        else:
            x = self.features(x)
            features = x.view(x.size(0), -1) # Flatten
        
        x = self.classifier(features)
        return x.unsqueeze(-1)
    


    """
    Shapes from Dataloader batch:
        RF_X shape: torch.Size([256, 2, 1024])
        RF_y shape: torch.Size([256])
        CFO_X shape: torch.Size([256, 2, 160])
        CFO_y shape: torch.Size([256, 1, 1])
        Channel_X shape: torch.Size([256, 2, 160])
        Channel_y shape: torch.Size([256, 2, 52])

--- Testing ChannelNet ---
ChannelNet model has 454248 parameters
Input shape for ChannelNet: torch.Size([256, 2, 160])
Output shape from ChannelNet: torch.Size([256, 2, 52])
--- ChannelNet Test End ---


--- Testing CFONetLarge ---
CFONetLarge model has 419137 parameters
Input shape for CFONetLarge: torch.Size([256, 2, 160])
Output shape from CFONetLarge: torch.Size([256, 1, 1])
--- CFONetLarge Test End ---


--- Testing CFONetSmall ---
CFONetSmall model has 697473 parameters
Input shape for CFONetSmall: torch.Size([256, 2, 160])
Output shape from CFONetSmall: torch.Size([256, 1, 1])
--- CFONetSmall Test End ---


--- Testing RFFingerprintingNet ---
RFFingerprintingNet model has 609744 parameters
Input shape for ChannelNet: torch.Size([256, 2, 1024])
Output shape from ChannelNet: torch.Size([256, 16])
--- RFFingerprintingNet Test End ---

    """