import pickle

# Path to the pickle file
pkl_path = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/cfo_partition_dict_0.5.pkl'
rf_pkl_path = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl'
channel_pkl_path = '/home/hofmann/Documents/projects/RepresentationLearning/dataset/channel_partition_dict_0.5.pkl'

# Load the pickle file
print(f"Loading data from {pkl_path}...")
with open(pkl_path, 'rb') as handle:
    data = pickle.load(handle)

with open(rf_pkl_path, 'rb') as handle:
    rf_data = pickle.load(handle)

with open(channel_pkl_path, 'rb') as handle:
    channel_data = pickle.load(handle)

print(rf_data['train'][0])
print(channel_data['train'][0])

# Print the contents of the loaded data
# print("Data loaded successfully!")
# print("Contents of the pickle file:")
# print(data)

from scipy.io import loadmat
#data['train'][0]

mat_data = loadmat(data['train'][0])
rf_mat_data = loadmat(rf_data['train'][0])
channel_mat_data = loadmat(channel_data['train'][0])






from pdb import set_trace
set_trace()

# You can also print more specific details, for example:
if isinstance(data, dict):
    for key, value in data.items():
        if hasattr(value, '__len__'):
            print(f"- Key: '{key}', Number of items: {len(value)}")
        else:
            print(f"- Key: '{key}', Value: {value}") 