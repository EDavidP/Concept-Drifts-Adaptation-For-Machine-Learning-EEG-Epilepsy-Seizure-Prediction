
def splitting(data_list, datetimes_list, seizure_onset_list, n_seizures_train, split_type):
     
    if split_type == 'Control partitioning':
    
        data = {}
        datetimes = {}
        seizure_onset = {}
        
        # Train        
        data['train'] = [data_list[:n_seizures_train]]
        datetimes['train'] = [datetimes_list[:n_seizures_train]]
        seizure_onset['train'] = [seizure_onset_list[:n_seizures_train]]
        
        # Test        
        data['test'] = data_list[n_seizures_train:]
        datetimes['test'] = datetimes_list[n_seizures_train:]
        seizure_onset['test'] = seizure_onset_list[n_seizures_train:]
        
    elif split_type == 'AddOneForgetOne': 
        
        n_seizures = len(data_list)
        
        data = {}
        datetimes = {}
        seizure_onset = {}
        
        # Train
        data['train'] = [data_list[package:package+n_seizures_train] for package in range(0,n_seizures-n_seizures_train)]
        datetimes['train'] = [datetimes_list[package:package+n_seizures_train] for package in range(0,n_seizures-n_seizures_train)]
        seizure_onset['train'] = [seizure_onset_list[package:package+n_seizures_train] for package in range(0,n_seizures-n_seizures_train)]
        
        # Test
        data['test'] = [data_list[package+n_seizures_train-1] for package in range(0,n_seizures-n_seizures_train)]
        datetimes['test'] = [datetimes_list[package+n_seizures_train-1] for package in range(0,n_seizures-n_seizures_train)]
        seizure_onset['test'] = [seizure_onset_list[package+n_seizures_train-1] for package in range(0,n_seizures-n_seizures_train)]
        
        
    elif split_type == 'Chronological': 
        
        n_seizures = len(data_list)
        
        data = {}
        datetimes = {}
        seizure_onset = {}
        
        # Train
        data['train'] = [data_list[:package+n_seizures_train] for package in range(0,n_seizures-n_seizures_train)]
        datetimes['train'] = [datetimes_list[:package+n_seizures_train] for package in range(0,n_seizures-n_seizures_train)]
        seizure_onset['train'] = [seizure_onset_list[:package+n_seizures_train] for package in range(0,n_seizures-n_seizures_train)]
        
        # Test
        data['test'] = [data_list[package+n_seizures_train-1] for package in range(0,n_seizures-n_seizures_train)]
        datetimes['test'] = [datetimes_list[package+n_seizures_train-1] for package in range(0,n_seizures-n_seizures_train)]
        seizure_onset['test'] = [seizure_onset_list[package+n_seizures_train-1] for package in range(0,n_seizures-n_seizures_train)]
  
            
    return data, datetimes, seizure_onset