import pickle
import numpy as np

from auxiliary_func import cut_data, construct_target, cut_sph, performance, remove_redundant_features
from regularization import get_firing_power, alarm_generation, alarm_processing
from evaluation import alarm_evaluation, sensitivity, FPR_h, statistical_validation
from plot_results import fig_test

def main_test_DWE(approach,split_type, patient, data_list, datetimes_list, seizure_onset_list, SPH, final_SOP_list, window_size):
    
    print('\nTesting...')

    # --- Load models ---  
    models = pickle.load(open(f'Models/{approach}/{split_type}/model_patient{patient}_{split_type}', 'rb'))  
   
    # --- Configurations ---
    SOP_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    n_seizures = len(data_list)
    firing_power_threshold = 0.5
    
    info_test = []
    for SOP in SOP_list:
            
        # --- Construct target (0 - interictal | 1 - preictal) ---
        target_list = [construct_target(datetimes_list[seizure], seizure_onset_list[seizure], SOP, SPH) for seizure in range(n_seizures)]

        # --- Cut/remove SPH --- Se um alarme não for disarado até SPH/10 minutos antes, é um falso alarme
        target_per_seizure = [cut_sph(target) for target in target_list]
        data_list = [data_list[seizure][:len(target_per_seizure[seizure])] for seizure in range(n_seizures)]
        datetimes_list = [datetimes_list[seizure][:len(target_per_seizure[seizure])] for seizure in range(n_seizures)]
            
        prediction_per_seizure = []
        alarm_per_seizure = []
        refractory_samples_per_seizure = []
        firing_power_per_seizure = []

        for seizure in range(n_seizures): 

            if split_type == 'GeneralSplit':

                # --- Model ---
                model = models[0][SOP]

            elif split_type == 'AddOneForgetOne' or split_type == 'SeizureAccumulation':

                # --- Model ---
                model = models[seizure][SOP]
            
            # --- Test ---
            prediction = test(data_list[seizure], model)

            # --- Regularization [Firing Power + Alarms] ---
            firing_power = get_firing_power(prediction, datetimes_list[seizure], SOP, window_size)
            alarm = alarm_generation(firing_power, firing_power_threshold)
            alarm, refractory_samples = alarm_processing(alarm, datetimes_list[seizure], SOP, SPH)

            firing_power_per_seizure.append(firing_power)
            prediction_per_seizure.append(prediction)
            alarm_per_seizure.append(alarm)
            refractory_samples_per_seizure.append(refractory_samples)
        

        # --- Concatenate seizures ---    
        target = np.concatenate(target_per_seizure)
        datetimes = np.concatenate(datetimes_list)
        prediction = np.concatenate(prediction_per_seizure)
        alarm = np.concatenate(alarm_per_seizure)
        refractory_samples = np.concatenate(refractory_samples_per_seizure)
        firing_power = np.concatenate(firing_power_per_seizure)
        
        # --- Performance [samples] ---
        ss_samples, sp_samples, metric, acc_samples = performance(target, prediction)
        
        # --- Performance [alarms] ---
        true_alarm, false_alarm = alarm_evaluation(target, alarm)
        ss = sensitivity(true_alarm, n_seizures)
        FPRh = FPR_h(target, false_alarm, refractory_samples, window_size)

        # --- Statistical validation ---
        surr_ss_mean, surr_ss_std, tt, pvalue = statistical_validation(target_per_seizure, alarm, ss, firing_power_threshold)

        # --- Save parameters & results ---
        info_test.append([ss_samples, sp_samples, firing_power_threshold, true_alarm, false_alarm, ss, FPRh, surr_ss_mean, surr_ss_std, tt, pvalue])
        print(f'\n--- TEST PERFORMANCE [SOP={SOP}] --- \nSS = {ss:.3f} | FPR/h = {FPRh:.3f}')
        print(f'--- Statistical validation ---\nSS surr = {surr_ss_mean:.3f} ± {surr_ss_std:.3f} (p-value = {pvalue:.4f})')

        # --- Figure: test ---
        fig_test(patient, SOP, target, prediction, firing_power, firing_power_threshold, alarm, 'Approach D', split_type)
        
    # for variable SOP
    if split_type == 'AddOneForgetOne' or split_type == 'SeizureAccumulation':
       
       # --- Construct target (0 - interictal | 1 - preictal) ---
       target_list = [construct_target(datetimes_list[seizure], seizure_onset_list[seizure], final_SOP_list[seizure], SPH) for seizure in range(n_seizures)]

       # --- Cut/remove SPH --- Se um alarme não for disarado até SPH/10 minutos antes, é um falso alarme
       target_per_seizure = [cut_sph(target) for target in target_list]
       data_list = [data_list[seizure][:len(target_per_seizure[seizure])] for seizure in range(n_seizures)]
       datetimes_list = [datetimes_list[seizure][:len(target_per_seizure[seizure])] for seizure in range(n_seizures)]

       prediction_per_seizure = []
       alarm_per_seizure = []
       refractory_samples_per_seizure = []
       firing_power_per_seizure = []
       
       for seizure in range(n_seizures): 

           # --- Model ---
           model = models[seizure][final_SOP_list[seizure]]
           
           # --- Test ---
           prediction = test(data_list[seizure], model)

           # --- Regularization [Firing Power + Alarms] ---
           firing_power = get_firing_power(prediction, datetimes_list[seizure], final_SOP_list[seizure], window_size)
           alarm = alarm_generation(firing_power, firing_power_threshold)
           alarm, refractory_samples = alarm_processing(alarm, datetimes_list[seizure], final_SOP_list[seizure], SPH)

           firing_power_per_seizure.append(firing_power)
           prediction_per_seizure.append(prediction)
           alarm_per_seizure.append(alarm)
           refractory_samples_per_seizure.append(refractory_samples)
       
       
       # --- Concatenate seizures ---    
       target = np.concatenate(target_per_seizure)
       datetimes = np.concatenate(datetimes_list)
       prediction = np.concatenate(prediction_per_seizure)
       alarm = np.concatenate(alarm_per_seizure)
       refractory_samples = np.concatenate(refractory_samples_per_seizure)
       firing_power = np.concatenate(firing_power_per_seizure)
       
       # --- Performance [samples] ---
       ss_samples, sp_samples, metric, acc_samples = performance(target, prediction)
       
       # --- Performance [alarms] ---
       true_alarm, false_alarm = alarm_evaluation(target, alarm)
       ss = sensitivity(true_alarm, n_seizures)
       FPRh = FPR_h(target, false_alarm, refractory_samples, window_size)

       # --- Statistical validation ---
       surr_ss_mean, surr_ss_std, tt, pvalue = statistical_validation(target_per_seizure, alarm, ss, firing_power_threshold)

       # --- Save parameters & results ---
       info_test.append([ss_samples, sp_samples, firing_power_threshold, true_alarm, false_alarm, ss, FPRh, surr_ss_mean, surr_ss_std, tt, pvalue])
       print(f'\n--- TEST PERFORMANCE [SOP={final_SOP_list}] --- \nSS = {ss:.3f} | FPR/h = {FPRh:.3f}')
       print(f'--- Statistical validation ---\nSS surr = {surr_ss_mean:.3f} ± {surr_ss_std:.3f} (p-value = {pvalue:.4f})')

       # --- Figure: test ---
       fig_test(patient, str(final_SOP_list), target, prediction, firing_power, firing_power_threshold, alarm, 'Approach A', split_type)
       
    print('\nTested\n\n')
        
    return info_test
    
    
def test(data, model):
    
    # --- Model ---
    scaler_list = model['scaler']
    selector_list = model['selector']
    svm_list = model['svm']
    ensemble_weights = model['ensemble_weights'] 
        
    # --- Ensemble classifiers ---
    predictions_list = []
    n_classifiers = len(svm_list)-1
        
    scaler = scaler_list[0]
    selector = selector_list[0]
        
    # --- Standardization ---
    data_test = scaler.transform(data)
    
    # --- Feature selection ---
    data_test = selector.transform(data_test)

    for classifier_i in range(n_classifiers):
        
        svm_model = svm_list[classifier_i]

        # --- Test ---
        prediction = svm_model.predict(data_test)
        predictions_list.append(prediction)
    
    # --- Classifiers Weighted vote ---
    predictions_array = np.transpose(np.array(predictions_list))
    ensemble_weights_array = np.transpose(np.array(ensemble_weights))

    final_prediction = np.matmul(predictions_array, ensemble_weights_array)   

    for i, prediction_ in enumerate(final_prediction):

        if prediction_ == 0.5:
            ensemble_weights_array_temp = ensemble_weights_array.copy()
            predictions_array_temp = predictions_array.copy()
            run = True
            while(run == True):
                if final_prediction[i] == 0.5:
                    ensemble_weights_array_temp = np.delete(ensemble_weights_array_temp, 0, axis=0) # delete the weight's row (oldest classifier)
                    predictions_array_temp = np.delete(predictions_array_temp, 0, axis=1) # delete prediction column 
                    final_prediction[i] = np.matmul(predictions_array_temp[i,:], ensemble_weights_array_temp)
                else:
                    run = False

    final_prediction = np.round_(final_prediction)

    return final_prediction