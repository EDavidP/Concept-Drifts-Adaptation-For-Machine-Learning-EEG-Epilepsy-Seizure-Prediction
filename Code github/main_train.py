from training_Control import main_train_Control
from training_BLW import main_train_BLW
from training_SbR import main_train_SbR
from training_DWE import main_train_DWE

def main_train(approach, split_type, patient, data, datetimes, seizure_onset, SPH, SOP_list):
    
    if approach=='Control': 
        main_train_Control(approach,split_type, patient, data, datetimes, seizure_onset, SPH, SOP_list)
    elif approach=='BLW': 
        main_train_SbR(approach,split_type, patient, data, datetimes, seizure_onset, SPH, SOP_list)
    elif approach=='SbR': 
        main_train_SbR(approach,split_type, patient, data, datetimes, seizure_onset, SPH, SOP_list)
    elif approach=='DWE': 
        main_train_DWE(approach,split_type, patient, data, datetimes, seizure_onset, SPH, SOP_list)
        