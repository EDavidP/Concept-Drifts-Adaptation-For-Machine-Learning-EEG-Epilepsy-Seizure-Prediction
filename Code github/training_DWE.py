import pickle
import numpy as np
import time as t

from auxiliary_func import remove_sop, get_sop, construct_target, cut_sph, standardization, select_features, classifier_weighted, get_weight_vector, performance_weighted, get_last_n_hours
from concept_dift_utils import grid_search_combination_list, windowing, get_ensemble_weights

def main_train_DWE(approach, split_type, patient, data_list, datetimes_list, seizure_onset_list, SPH, SOP_list):
    
    # --- Configurations ---
    k = np.arange(start = 10, stop = 41, step = 10) # number of features
    c_pot = np.arange(start = -10, stop = 11, step = 2, dtype=float) 
    C = 2**c_pot # parameter of SVM classifier
    window_size_minutes = 60
    
    
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
            best_k, best_C, ss, sp, best_metric = grid_search(patient, data, target, datetimes, seizure_onset_list[i], k, C, SPH, SOP, feature_selction_algorithm, window_size_minutes)
            # --- Train --- 
            model = train(data, target, datetimes, seizure_onset_list[i], best_k, best_C, SPH, SOP, feature_selction_algorithm, window_size_minutes)
            
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
    
    

def grid_search(patient, data, target_list, datetimes_list, seizure_onset_list, k, C, SPH, SOP, feature_selction_algorithm, window_size_minutes):
    
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
                target_validation = target_list[fold_i]
                train_list = [data[fold_j] for fold_j in combination[0]]
                target_train_list = [target_list[fold_j] for fold_j in combination[0]]
                datetimes_train_list = [datetimes_list[fold_j] for fold_j in combination[0]]

                set_train = np.concatenate(train_list)
                target_train = np.concatenate(target_train_list)

                # --- Standardization ---
                scaler = standardization(set_train)
                set_train = scaler.transform(set_train)
                train_list = [scaler.transform(train_seizure) for train_seizure in train_list]
                set_validation = scaler.transform(set_validation)

                # --- Feature selection ---
                selector = select_features(set_train, target_train, k_i, feature_selction_algorithm)      
                train_list = [selector.transform(train_seizure) for train_seizure in train_list]
                set_validation = selector.transform(set_validation) 

                # --- Get last 2 hours from last train seizure---
                idx_last_2_hours = get_last_n_hours(2, datetimes_train_list[-1], seizure_onset_list[fold_i-1], SPH)
                train_last_2_hours = train_list[-1][idx_last_2_hours]
                target_train_last_2_hours = target_train_list[-1][idx_last_2_hours]
                
                # --- Store the SOPs 
                idx_sop_list = [get_sop(target_train) for target_train in target_train_list]
                sop_data_list = [train_list[seizure][idx_sop_list[seizure]] for seizure in range(len(idx_sop_list))]

                # --- Remove SOP ---
                target_list_ = [remove_sop(target) for target in target_list]
                train_list_ = [train_list[seizure][:len(target_list_[seizure])] for seizure in range(len(idx_sop_list))]
                datetimes_train_list_ = [datetimes_train_list[seizure][:len(target_list_[seizure])] for seizure in range(len(idx_sop_list))]

                svm_list = []
                predictions_list = []
                for seizure_idx in range(len(target_train_list)):

                    windows_dict = windowing(datetimes_train_list_[seizure_idx], window_size_minutes)
                    for key, idx in windows_dict.items():

                        segment = train_list_[seizure_idx][idx,:]
                        target_segment = target_train_list[seizure_idx][idx]
                        
                        if segment.shape[0] == 0:
                            continue
                    
                        segment = np.concatenate((segment,sop_data_list[seizure_idx]), axis=0)
                        target_segment = np.concatenate((target_segment,np.ones(len(sop_data_list[seizure_idx]))), axis=0)

                        weight_vector = get_weight_vector(target_segment)

                        # --- Train (training set) ---
                        svm_model = classifier_weighted(segment, target_segment, weight_vector, C_i, 'svm')
                        svm_list.append(svm_model)

                        # --- Test (validation set) ---
                        prediction = svm_model.predict(set_validation)
                        predictions_list.append(prediction)


                # --- Weighted vote ---
                predictions_array = np.transpose(np.array(predictions_list))
                ensemble_weights = get_ensemble_weights(svm_list, train_last_2_hours, target_train_last_2_hours)
                ensemble_weights_array = np.transpose(np.array(ensemble_weights))
                
                prediction_validation = np.matmul(predictions_array, ensemble_weights_array)

                for i, prediction_ in enumerate(prediction_validation):

                    if prediction_ == 0.5: #tie
                        ensemble_weights_array_temp = ensemble_weights_array.copy()
                        predictions_array_temp = predictions_array.copy()
                        run = True
                        while(run == True):
                            if prediction_validation[i] == 0.5:
                                ensemble_weights_array_temp = np.delete(ensemble_weights_array_temp, 0, axis=0) # delete the weight's row (oldest classifier)
                                predictions_array_temp = np.delete(predictions_array_temp, 0, axis=1) # delete prediction column 
                                prediction_validation[i] = np.matmul(predictions_array_temp[i,:], ensemble_weights_array_temp)
                            else:
                                run = False
                
                weight_vector_validation = get_weight_vector(target_validation)

                prediction_validation = np.round_(prediction_validation)

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



def train(data_list, target_list, datetimes_list, seizure_onset_list, best_k, best_C, SPH, SOP, feature_selction_algorithm, window_size_minutes):

    print('\n\nTraining classifier...')
    start_time = t.time()

    scaler_list = []
    # redundant_features_list = []
    selector_list = []
    svm_list = []

    data_train = np.concatenate(data_list)
    target_train = np.concatenate(target_list)

    # --- Standardization ---
    scaler = standardization(data_train)
    data_train = scaler.transform(data_train)
    data_list = [scaler.transform(data_seizure) for data_seizure in data_list]

    # --- Feature selection ---
    selector = select_features(data_train, target_train, int(best_k), feature_selction_algorithm)  
    data_train = selector.transform(data_train)    
    data_list = [selector.transform(data_seizure_) for data_seizure_ in data_list]

    # --- Get last 2 hours from last seizure---
    idx_last_2_hours = get_last_n_hours(2, datetimes_list[-1], seizure_onset_list[-1], SPH)
    data_last_2_hours = data_list[-1][idx_last_2_hours]
    target_last_2_hours = target_list[-1][idx_last_2_hours]
    
    # --- Store the SOPs 
    idx_sop_list = [get_sop(target) for target in target_list]
    sop_data_list = [data_list[seizure][idx_sop_list[seizure]] for seizure in range(len(data_list))]

    # --- Remove SOP ---
    target_list = [remove_sop(target_i) for target_i in target_list]
    data_list = [data_list[seizure][:len(target_list[seizure])] for seizure in range(len(data_list))]
    datetimes_list = [datetimes_list[seizure][:len(target_list[seizure])] for seizure in range(len(data_list))]

    svm_list = []
    for seizure_idx in range(len(data_list)):

        windows_dict = windowing(datetimes_list[seizure_idx], window_size_minutes)
        for key, idx in windows_dict.items():

            segment = data_list[seizure_idx][idx,:]
            target_segment = target_list[seizure_idx][idx]
            
            if segment.shape[0] == 0:
                continue
        
            segment = np.concatenate((segment,sop_data_list[seizure_idx]), axis=0)
            target_segment = np.concatenate((target_segment,np.ones(len(sop_data_list[seizure_idx]))), axis=0)

            weight_vector = get_weight_vector(target_segment)

            # --- Train (training set) ---
            svm_model = classifier_weighted(segment, target_segment, weight_vector, best_C, 'svm')
            svm_list.append(svm_model)


    ensemble_weights = get_ensemble_weights(svm_list, data_last_2_hours, target_last_2_hours)

    # --- Save --- 
    scaler_list.append(scaler)
    selector_list.append(selector)
    svm_list.append(svm_model)

    model = {}
    model['scaler'] = scaler_list
    model['selector'] = selector_list
    model['svm'] = svm_list
    model['ensemble_weights'] = ensemble_weights

    end_time = t.time()
    run_time = end_time - start_time
    print(f'running time: {run_time:.2f}')
    print('Classifier trained')

    return model