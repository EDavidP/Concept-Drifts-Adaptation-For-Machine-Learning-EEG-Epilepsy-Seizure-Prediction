import pickle
import numpy as np
import time as t

from auxiliary_func import cut_data, construct_target, cut_sph, standardization, select_features, get_weight_vector, classifier_weighted, performance_weighted
from concept_dift_utils import window_adjustment, grid_search_combination_list

def main_train_BLW(approach,split_type, patient, data_list, datetimes_list, seizure_onset_list, SPH, SOP_list):
    
    # --- Configurations ---
    k = np.arange(start = 10, stop = 41, step = 10) # number of features
    c_pot = np.arange(start = -10, stop = 11, step = 2, dtype=float) 
    C = 2**c_pot # parameter of SVM classifier
    window_step_minutes = 60
    
    feature_selction_algorithm = 'ANOVA'
    # feature_selction_algorithm = 'Random Forest Ensemble'

    models = {}
    for i in range(len(data_list)): # iterations
        print(f'\n\nTraining iteration #{i+1}\n')

        models_SOP = {}
        for SOP in SOP_list:

            n_seizures = len(data_list[i])
            # --- Construct target (0 - interictal | 1 - preictal) ---
            target_list = [construct_target(datetimes_list[i][seizure], seizure_onset_list[i][seizure], SOP, SPH) for seizure in range(n_seizures)]

            # --- Cut SPH ---
            target = [cut_sph(target) for target in target_list]
            data = [data_list[i][seizure][:len(target[seizure])] for seizure in range(n_seizures)]
            datetimes = [datetimes_list[i][seizure][:len(target[seizure])] for seizure in range(n_seizures)]
            
            # --- Grid search ---
            best_k, best_C, ss, sp, best_metric = grid_search(patient, data, target, datetimes, k, C, SPH, SOP, window_step_minutes, feature_selction_algorithm)
                                           
            # --- Train --- 
            model = train(data, target, datetimes, best_k, best_C, SOP, window_step_minutes, feature_selction_algorithm)
            
            # --- Save parameters & results ---
            models_SOP[SOP] = model
            models_SOP[SOP]['info_train'] = [SOP, best_k, best_C, ss, sp, best_metric]
            
    
        # --- Save parameters & results ---
        models[i] = models_SOP
        models['split_type'] = split_type
        info_train = [models_SOP[SOP]['info_train'] for SOP in SOP_list]
        metrics = [info[5] for info in info_train] # metric (Classifier)
        idx_best = np.argmax(metrics)
        models_SOP['iteration_info_train'] = info_train[idx_best]
     
    # --- Save models ---
    pickle.dump(models, open(f'Models/{approach}/{split_type}/model_patient{patient}_{split_type}', 'wb'))    

            

def grid_search(patient, data, target, datetimes_list, k, C, SPH, SOP, window_step_minutes, feature_selction_algorithm):

    print('\nGrid search...')
    
    # --- Cross validation ----
    n_folds = len(data) # the number of folds is the number of seizures
    
    performances = []
    for k_i in k:            
        for C_i in C:
            print(f'\n--------- k = {k_i} | C = {C_i:.2g} ---------')
            start_time_combination = t.time() # to compute time that each combination k,C lasts
    
            ss_per_combination = []
            sp_per_combination = []
            metric_per_combination = []            
            combination_list_ = grid_search_combination_list(n_folds)
            for combination in combination_list_:
                """ 0,1->2|1->2|0->1 """
                    
                # --- Data splitting (train & validation) ---
                fold_i = combination[1][0]

                set_validation = data[fold_i]
                set_train = [data[fold_j] for fold_j in range(n_folds) if fold_j!=fold_i]
                datetimes_train = [datetimes_list[fold_j] for fold_j in range(n_folds) if fold_j!=fold_i]
                target_validation = target[fold_i]
                target_train = [target[fold_j] for fold_j in range(n_folds) if fold_j!=fold_i]
                
                set_train = np.concatenate(set_train)
                target_train = np.concatenate(target_train)
                datetimes = np.concatenate(datetimes_train)

                # --- Standardization ---
                scaler = standardization(set_train)
                set_train = scaler.transform(set_train)
                set_validation = scaler.transform(set_validation)
                
                # --- Feature selection ---
                selector = select_features(set_train, target_train, k_i, feature_selction_algorithm)      
                set_train = selector.transform(set_train)
                set_validation = selector.transform(set_validation) 
                
                weight_vector_train = get_weight_vector(target_train)
                weight_vector_validation = get_weight_vector(target_validation)

                # --- Train (training set) ---
                svm_model = classifier_weighted(set_train, target_train, weight_vector_train, C_i, 'svm')
                
                # --- Test (validation set) ---
                prediction_validation = svm_model.predict(set_validation)
                    
                # --- Performance ---
                ss, sp, metric, acc = performance_weighted(target_validation, prediction_validation, weight_vector_validation)
                
                ss_per_combination.append(ss)
                sp_per_combination.append(sp)
                metric_per_combination.append(metric)
                    
            ss_avg = np.mean(ss_per_combination)
            sp_avg = np.mean(sp_per_combination)
            metric_avg = np.mean(metric_per_combination)
            print(f'Average performance: SS = {ss_avg:.2f} | SP = {sp_avg:.2f} | metric: {metric_avg:.2f}')
    
            end_time_combination = t.time()
            run_time = end_time_combination - start_time_combination
            print(f'Running time per combination = {run_time:.2f}')
            
            performances.append([k_i, C_i, ss_avg, sp_avg, metric_avg, run_time, patient, n_folds, SPH, SOP])
            
    performances = np.array(performances).astype(float)
        
    # --- Select best parameters ---
    # Best performance (maximum metric)
    best_performance = max(performances[:,4])
    idx_best = np.where(performances[:,4] == best_performance)[0] # get array of indexes
    # Tiebreaker (minimum running time)          
    if len(idx_best)>1:
        tiebreaker_performance = min(performances[idx_best,5])
        idx_best_tie = np.where(performances[idx_best,5] == tiebreaker_performance)[0][0] # get index
        idx_best = idx_best[idx_best_tie]
    else:
        idx_best = idx_best[0]
    
    # --- Save selected parameters & results ---
    best_k = performances[idx_best,0]
    best_C = performances[idx_best,1]
    ss = performances[idx_best,2]
    sp = performances[idx_best,3]
    best_metric = performances[idx_best,4]  # best_performance
    
    print('\nGrid search completed')
    print(f'\n --------------- GRID SEARCH (best result) --------------- \nk = {best_k:.2f} | C = {best_C:.2f} | SS = {ss:.2f} | SP = {sp:.2f} | metric = {best_metric:.2f}')
        
    return best_k, best_C, ss, sp, best_metric
                    

def train(data, target, datetimes, best_k, best_C, SOP, window_step_minutes, feature_selction_algorithm):
    
    print('\n\nTraining classifier...')
    start_time = t.time()

    scaler_list = []
    selector_list = []
    svm_list = []
        
    data_train = np.concatenate(data)
    target_train = np.concatenate(target)
    datetimes_train = np.concatenate(datetimes)
    
    # --- Standardization ---
    scaler = standardization(data_train)
    data_train = scaler.transform(data_train)
    
    # --- Feature selection ---
    selector = select_features(data_train, target_train, int(best_k), feature_selction_algorithm)      
    data_train = selector.transform(data_train)

    # --- Estimate window size with SVM ---
    window_size_minutes = window_adjustment(data_train, target_train, datetimes_train, SOP, window_step_minutes)
                
    # --- Cut data ---
    idx = cut_data(datetimes_train, window_size_minutes)
    target_train = target_train[idx]
    data_train = data_train[idx]

    weight_vector = get_weight_vector(target_train)
    
    # --- Train (training set) ---
    svm_model = classifier_weighted(data_train, target_train, weight_vector, best_C, 'svm')
    
    # --- Save --- 
    scaler_list.append(scaler)
    selector_list.append(selector)
    svm_list.append(svm_model)
                    
    # --- Model ---
    model = {}
    model['scaler'] = scaler_list
    model['selector'] = selector_list
    model['svm'] = svm_list
    model['window_size'] = window_size_minutes

    end_time = t.time()
    run_time = end_time - start_time
    print(f'Window size: {window_size_minutes} minutes')
    print(f'running time: {run_time:.2f}')
    print('Classifier trained')

    return model