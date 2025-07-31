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
            activation_dir (string): Directory with all the activation .pt files.
            file_list (list): List of filenames to include in this dataset slice.
            test_mode (bool): In test mode, we don't need to load all data, just what's needed for one sample.
        """
        self.activation_dir = activation_dir
        self.file_list = list(file_list)  # Create a mutable copy
        self.test_mode = test_mode

        if not os.path.isdir(self.activation_dir):
            raise ValueError(f"Activation directory not found: {self.activation_dir}")

        # For training mode, shuffle the list of files to ensure random batches
        if not self.test_mode:
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

        # Check if we're dealing with fixed RF data (3 slices per sample)
        # We'll check the first available activation file to determine the structure
        first_activation_file = next(iter(self.activation_map.values()))
        sample_data = torch.load(first_activation_file, map_location='cpu')
        activation_shape = sample_data['activation'].shape
        
        # If activation has 3 dimensions and first dim is 3, it's from fixed RF data
        self.is_fixed_rf = len(activation_shape) == 3 and activation_shape[0] == 3
        
        if self.is_fixed_rf:
            print(f"Detected fixed RF data format: activations have shape {activation_shape}")
            # For fixed RF, we need to expand the dataset to handle 3 slices per sample
            self.expanded_file_list = []
            for mat_file in self.file_list:
                base_filename = os.path.basename(mat_file)
                if base_filename in self.activation_map:
                    # Add 3 entries for each file (one for each slice)
                    for slice_idx in range(3):
                        self.expanded_file_list.append((mat_file, slice_idx))
        else:
            print(f"Detected standard data format: activations have shape {activation_shape}")
            self.expanded_file_list = [(mat_file, 0) for mat_file in self.file_list if os.path.basename(mat_file) in self.activation_map]

        print(f"Initialized dataset with {len(self.file_list)} files. Found and mapped {len(self.activation_map)} activations.")
        print(f"Total samples after expansion: {len(self.expanded_file_list)}")

    def __len__(self):
        return len(self.expanded_file_list)
    
    def __getitem__(self, index):
        """
        Returns a tuple containing all data for one sample:
        (rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, filename)
        """
        mat_file_path, slice_idx = self.expanded_file_list[index]
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
        
        # Handle fixed RF data: select the appropriate slice
        if self.is_fixed_rf:
            # For fixed RF, activation has shape (3, 2, 1024) - select the appropriate slice
            if activation.shape[0] > slice_idx:
                activation = activation[slice_idx:slice_idx+1]  # Keep as (1, 2, 1024)
            else:
                # Handle case where we don't have enough slices
                activation = activation[0:1]  # Use first slice as fallback
        
        #print(f"[py_datasets.py->__getitem__] Loaded data for {original_filename}. Shapes: RF_X={rf_x.shape}, CFO_X={cfo_x.shape}, Channel_X={channel_x.shape}, activation={activation.shape}")

        return rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, original_filename 