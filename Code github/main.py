import os
import pickle
import math

os.chdir(r"C:\Users\edson\Desktop\UC\2021-2022\Projeto\Code")

# os.chdir(r"C:\Users\mauro\OneDrive\Ambiente de Trabalho\Edson\Code")

# Scripts
from import_data import import_features, import_datetimes, read_info
from splitting import splitting
from main_train import main_train
from main_test import main_test
from save_results import select_final_result, save_results, save_final_results
from plot_results import fig_performance, fig_performance_per_patient, fig_final_performance, fig_parameters_selection



# %% CONFIGURATIONS

TrainOrTest = 'TRAIN'
# TrainOrTest = 'TEST'

# --- Data split option ---
# split_type = 'Control partitioning'
split_type = 'AddOneForgetOne'
# split_type = 'Chronological'

# --- Approach option ---
approach = 'Control' # Control 
# approach = 'BLW' 
# approach = 'SbR'
# approach = 'DWE'

# --- Train options ---
n_seizures_train = 3
SPH = 10
SOP_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
window_size = 5 # 5 segundos firing power

channels = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','FZ','CZ','PZ']
f = open("Data/univariate_feature_names.txt", "r")
features = []
for x in f:
    features.append(x)

print(f'\n\n==================== {approach.upper()} ====================')

# %% IMPORT DATA

information_general = {}
information_train = {}
information_test = {}
final_information = []
angle_results = {}
info_angle_results = {}
info_window_size = {}

gap_between_seizures = []
seizure_duration = {}

patient_IDs =  ['30802']                

directory = 'Data'
for patient in patient_IDs:

    folder='pat_'+str(patient)+'_features'
    path = os.path.join(directory,folder)
    print(f'\n\n----- Patient {patient} -----')

    data_list = []
    datetimes_list = [] 
    path1 = os.path.join(path,'features')
    for file in sorted(os.listdir(path1), key=len): # each file corresponds to a seizure [key=len to sort by natural order (1,2,...,11 instead of 1,11,...2)]
        file_path = os.path.join(path1,file)
        filename = file[:file.index('.')]
        seizure = filename.split('_')[3]
        print(f'\n\n- features seizure {seizure} -')
        
        # --- Import features ---
        data = import_features(file_path)
        data_list.append(data)
        
    path2 = os.path.join(path,'datetimes')
    for file in sorted(os.listdir(path2), key=len): # each file corresponds to a seizure [key=len to sort by natural order (1,2,...,11 instead of 1,11,...2)]
        file_path = os.path.join(path2,file)
        filename = file[:file.index('.')]
        seizure = filename.split('_')[2]
        print(f'\n\n- datetimes seizure {seizure} -')
        
        # --- Import datetimes ---
        datetimes = import_datetimes(file_path)
        datetimes_list.append(datetimes)

    
    # --- Import seizures information --- the onset of each seizure
    path = os.path.join(path,'all_seizure_information.pkl')
    seizure_onset_list = read_info(path)

    

# %% SPLIT into TRAIN | TEST
    
    # --- Splitting data ---
    data, datetimes, seizure_onset = splitting(data_list, datetimes_list, seizure_onset_list, n_seizures_train,split_type)
    
# %% TRAIN

    if TrainOrTest == 'TRAIN':    
        # --- Train --- 
        Dict = main_train(approach, split_type, patient, data['train'], datetimes['train'], seizure_onset['train'], SPH, SOP_list)
        
    
# %% TEST

    if TrainOrTest == 'TEST':  
        # --- Test --- 
        models = pickle.load(open(f'Models/{approach}/{split_type}/model_patient{patient}_{split_type}', 'rb'))  
        
        n_iterations = len(models)-1
        final_SOP_list = [models[iteration]['iteration_info_train'][0] for iteration in range(0,n_iterations)]
        
        info_train = [models[0][SOP]['info_train'] for SOP in SOP_list] # first iteration info_train
        info_test = main_test(approach, split_type, patient, data['test'], datetimes['test'], seizure_onset['test'], SPH, final_SOP_list, window_size)
        
        # --- Select & save final result ---
        info_general = [patient, n_seizures_train, len(data['test']), SPH]
        
        
        idx_final = select_final_result(info_train, approach)
        
        if split_type == 'Control partitioning':

            final_information.append(info_general + info_train[idx_final] + info_test[idx_final])

        elif split_type == 'AddOneForgetOne' or split_type == 'Chronological':
            
            # add info_train for the variable SOP
            final_best_k_list = [models[iteration]['iteration_info_train'][1] for iteration in range(0,n_iterations)]
            final_best_C_list = [int(math.log(models[iteration]['iteration_info_train'][2],2)) for iteration in range(0,n_iterations)]
            final_SS_list = [round(models[iteration]['iteration_info_train'][3],2) for iteration in range(0,n_iterations)]
            final_SP_list = [round(models[iteration]['iteration_info_train'][4],2) for iteration in range(0,n_iterations)]
            final_best_metric_list = [models[iteration]['iteration_info_train'][5] for iteration in range(0,n_iterations)]
            info_train.append([str(final_SOP_list), str(final_best_k_list), str(final_best_C_list), str(final_SS_list), str(final_SP_list), str(final_best_metric_list)])
            
            final_information.append(info_general + info_train[-1] + info_test[-1])
        
        
        info_train[idx_final][0] = f'{info_train[idx_final][0]}*' # mark final result (first iteration)
        
        # --- Save results (patient) ---
        information_general[patient] = info_general
        information_train[patient] = info_train
        information_test[patient] = info_test
        
        # --- Figure: performance (patient) ---
        fig_performance(patient, info_train, info_test, approach, split_type)

    
# %% SAVE FINAL RESULS
if TrainOrTest == 'TEST':
    # --- Save results (excel) ---
    save_results(information_general, information_train, information_test, approach, split_type)
    
    # --- Figure: performance per patient (all SOPs) ---
    fig_performance_per_patient(information_test, approach, split_type)
    
    # --- Save final results (excel) ---
    save_final_results(final_information, approach ,split_type)
    
    # --- Figure: final performance per patient (selected SOPs) ---
    fig_final_performance(final_information, approach ,split_type)
    
    # --- Figure: selected features ---
    fig_parameters_selection(final_information, approach, split_type, channels, features)

