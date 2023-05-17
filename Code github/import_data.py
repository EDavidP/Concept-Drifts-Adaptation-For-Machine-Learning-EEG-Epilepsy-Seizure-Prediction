import numpy as np
import pickle
from datetime import datetime as dt

def import_features(path):
    
    print('\nLoading features...')
    
    features_per_seizure=np.load(path)
    number_of_chanels = features_per_seizure.shape[2]
    number_of_features = features_per_seizure.shape[1]
    data_seizure = []
    
    for idx in range(number_of_features):
        channels_per_feature = features_per_seizure[:,idx,:]
        if idx==0:
            data_seizure = channels_per_feature
        else:
            data_seizure = np.concatenate((data_seizure,channels_per_feature),axis=1)
            
    print('features loaded')
    
    return data_seizure

def import_datetimes(path):
    
    print('\nLoading datetimes...')
    
    timestamps_per_seizure=np.load(path)
    datetimes = [dt.fromtimestamp(timestamps_per_seizure[idx]) for idx in range(len(timestamps_per_seizure))]
    datetimes=np.array(datetimes)
    
    print('datetimes loaded')
    
    return datetimes
    
def read_info(path):
    
    print('\nLoading seizure onset datetimes...')
    
    with open(path, 'rb') as info:
        seizures_info = pickle.load(info)
    
    timestamps = []
    for seizure_idx in range(len(seizures_info)):
        seizure_begin = seizures_info[seizure_idx][0]
        timestamps.append(seizure_begin)
        
    selected_info = [dt.fromtimestamp(float(timestamps[idx])) for idx in range(len(timestamps))]
    print('seizure onset datetimes loaded.')
    
    return selected_info