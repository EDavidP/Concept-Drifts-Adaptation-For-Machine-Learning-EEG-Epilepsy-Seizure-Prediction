from testing_Control import main_test_Control
from testing_BLW_SbR import main_test_BLW_SbR
from testing_DWE import main_test_DWE

def main_test(approach, split_type, patient, data, times, metadata, SPH, final_SOP_list, window_size):
    
    if approach=='Control': 
        info_test = main_test_Control(approach,split_type, patient, data, times, metadata, SPH, final_SOP_list, window_size)
    elif approach=='BLW' or approach=='SbR':
        info_test = main_test_BLW_SbR(approach, split_type, patient, data, times, metadata, SPH, final_SOP_list, window_size)
    elif approach=='DWE':
        info_test = main_test_DWE(approach,split_type, patient, data, times, metadata, SPH, final_SOP_list, window_size)
    return info_test

