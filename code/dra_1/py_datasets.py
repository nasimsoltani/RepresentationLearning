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
        self.expanded_file_list = []  # Will contain individual slice files for RF Fixed datasets
        
        for mat_path in self.file_list:
            base_name = os.path.basename(mat_path).replace('.mat', '.pth')
            
            # Check for direct mapping first (regular datasets)
            if base_name in available_activations:
                self.activation_map[os.path.basename(mat_path)] = os.path.join(self.activation_dir, base_name)
                self.expanded_file_list.append(mat_path)
            else:
                # Check for slice-based files (RF Fixed datasets)
                slice_files_found = []
                for slice_idx in range(3):  # Check for up to 3 slices
                    slice_name = base_name.replace('.pth', f'_slice{slice_idx}.pth')
                    if slice_name in available_activations:
                        slice_mat_name = os.path.basename(mat_path).replace('.mat', f'_slice{slice_idx}.mat')
                        self.activation_map[slice_mat_name] = os.path.join(self.activation_dir, slice_name)
                        self.expanded_file_list.append(mat_path.replace('.mat', f'_slice{slice_idx}.mat'))
                        slice_files_found.append(slice_idx)
                
                # If no slices found, check if the original file exists
                if not slice_files_found and base_name not in available_activations:
                    print(f"Warning: No activation files found for {os.path.basename(mat_path)} (neither direct nor slice-based)")

        # Update file_list to use the expanded list that includes slice files
        self.file_list = self.expanded_file_list
        
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