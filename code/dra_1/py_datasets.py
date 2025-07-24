import os
import torch
from torch.utils.data import Dataset
import random


class ActivationDataset(Dataset):
    """
    Dataset class that reads .pth activation files corresponding to a given list of .mat files.
    
    Each .pth file, saved by extract_activations.py, contains:
    - 'activation': encoder activations
    - 'filename': original .mat filename
    - 'RF_X', 'CFO_X', 'Channel_X': input data
    - 'rf_label', 'cfo_label', 'channel_label': labels
    """
    
    def __init__(self, activation_dir, file_list, test_mode=False):
        """
        Args:
            activation_dir (str): Directory containing pre-computed .pth activation files.
            file_list (list): List of .mat file paths (from a partition dictionary)
                              that determines which activations to load.
            test_mode (bool): If False, the file_list is shuffled.
        """
        self.activation_dir = activation_dir
        self.file_list = file_list.copy()  # Make a copy to avoid modifying the original
        self.test_mode = test_mode

        if not os.path.isdir(self.activation_dir):
            raise ValueError(f"Activation directory not found: {self.activation_dir}")

        # Always shuffle to ensure class diversity, regardless of test_mode
        # This is important because files are often ordered by class/radio ID
        random.shuffle(self.file_list)

        # Create a set of available activation file basenames for quick lookups
        available_activations = {os.path.basename(f) for f in os.listdir(self.activation_dir) if f.endswith('.pth')}
        
        # Build the activation map based on the provided file_list
        self.activation_map = {}
        for mat_path in self.file_list:
            base_name = os.path.basename(mat_path).replace('.mat', '.pth')
            if base_name in available_activations:
                self.activation_map[os.path.basename(mat_path)] = os.path.join(self.activation_dir, base_name)

        if not self.activation_map:
            raise ValueError(f"No '.pth' activation files in {self.activation_dir} correspond to the files in file_list.")

        print(f"Initialized dataset with {len(self.file_list)} files. Found and mapped {len(self.activation_map)} activations.")

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        Returns a tuple containing all data for one sample:
        (rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, filename)
        """
        mat_file_path = self.file_list[index]
        base_filename = os.path.basename(mat_file_path)

        # Find the corresponding .pth file from our map
        pth_file_path = self.activation_map.get(base_filename)
        
        if pth_file_path is None:
            # This handles cases where activation is missing for a .mat file in the list.
            raise FileNotFoundError(
                f"Could not find corresponding .pth activation for {base_filename} "
                f"in {self.activation_dir}"
            )

        # Load the data from the .pth file
        data = torch.load(pth_file_path, map_location='cpu')
        
        # Extract all data parts
        rf_x = data['RF_X']
        rf_y = data['rf_label']
        cfo_x = data['CFO_X']
        cfo_y = data['cfo_label']
        channel_x = data['Channel_X']
        channel_y = data['channel_label']
        activation = data['activation']
        original_filename = data['filename']
        
        #print(f"[py_datasets.py->__getitem__] Loaded data for {original_filename}. Shapes: RF_X={rf_x.shape}, CFO_X={cfo_x.shape}, Channel_X={channel_x.shape}, activation={activation.shape}")

        return rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, original_filename 