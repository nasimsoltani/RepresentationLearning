import pickle

with open('/scratch/10608/aadharsh_aadhithya/data/rep_lr/OracleDatasetProcessed-arranged/rf_partition_dict_0.5.pkl', 'rb') as f:
    data = pickle.load(f)


print(data.keys())
print(data['max_cfo'])
print(data['train'])