import pickle

with open('/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['max_cfo'])