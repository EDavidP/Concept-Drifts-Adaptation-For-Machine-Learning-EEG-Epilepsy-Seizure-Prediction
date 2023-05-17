
from logging import NullHandler
import numpy as np
from math import floor
from math import pi
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from datetime import timedelta
import time as t
from scipy import stats

from sklearn import preprocessing, feature_selection, svm, metrics, model_selection
from auxiliary_func import cut_data, construct_target, cut_sph, class_balancing, standardization, find_redundant_features, remove_redundant_features, select_features, get_weight_vector, classifier_weighted, performance ,performance_weighted


#SVM based method
def window_adjustment(data, target, datetimes, SOP, window_step_minutes):
    # the SPH is already cut!
    # the initial window will be the size of SOP*2 and increment every second
    # the crises come concatenated

    print('\nWindow adjustment...')
    tolerance = 12 # 12 hours
    start_time = t.time()

    window_size_list = []
    estimate_list = []

    Dict = incremental_windowing(datetimes, window_step_minutes, SOP)
    # print(Dict)
    

    for key, window_idx in Dict.items():
        if len(window_idx) >= 100: 

            # print('\nCalculating LOOE...')
            start_time_2 = t.time()

            begin_window = window_idx[-1] # it's upside down, because windowing is done backwards
            end_window = window_idx[0]
            window_size_minutes = floor((datetimes[end_window] - datetimes[begin_window]).total_seconds() / 60.0)

            X_i = data[window_idx]
            Y_i = target[window_idx]
            
            estimate = leave_one_out_estimate(X_i, Y_i)
            
            window_size_list.append(window_size_minutes)
            estimate_list.append(estimate)

            end_time_2 = t.time()
            run_time_2 = end_time_2 - start_time_2
            # print(f'\nrunning time: {run_time_2:.2f}')
            # print('Calculating LOOE completed.')

            if key > tolerance: 
                if early_stoping(estimate_list, tolerance) == True:
                    break


    min_estimate = min(estimate_list)
    idx = estimate_list.index(min_estimate)
    final_window_size = window_size_list[idx] # size in minutes
    '''print(type(final_window_size))
    print(type(np.int32(final_window_size).item()))'''
    final_window_size = np.int32(final_window_size).item()

    end_time = t.time()
    run_time = end_time - start_time
    print(f'window size: {final_window_size}')
    print(f'running time: {run_time:.2f}')
    print('Window adjustment completed')
    
    return final_window_size



def incremental_windowing(datetimes, window_step_minutes, SOP):
    """ Returns the indexes of each new window """
    
    Dict = {}
    window_idx = 0
    window_idx_list = []

    begin_window = datetimes[-1] - 2*timedelta(minutes = SOP) # the first window starts here

    double_SOP_idx_list = []
    for idx in  range(len(datetimes)-1,-1,-1):

        if datetimes[idx] >= begin_window:
            double_SOP_idx_list.append(idx)
        else:
            break

    double_SOP_idx_array = np.array(double_SOP_idx_list).astype('int')

    begin = double_SOP_idx_array[-1]-1

    for i in range(len(datetimes)-1,-1,-1):
        """ This does the windowing backwards """

        if datetimes[i] < ( datetimes[begin] - timedelta(minutes = window_step_minutes)): # if outside the window
            window_idx_array = np.array(window_idx_list).astype('int') 
            if window_idx == 0:
                Dict[window_idx] = double_SOP_idx_array
            else:
                Dict[window_idx] = np.concatenate((Dict[window_idx - 1],window_idx_array), axis=0) 
            
            begin = i
            window_idx_list = []
            window_idx = window_idx + 1
            
        else:
            window_idx_list.append(i)
            
    return Dict

def early_stoping(estimate_list, tolerance):

    x = [x_i for x_i in range(tolerance)]
    y = estimate_list[-tolerance:]

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    if slope >= 0.05: # stop window_adjustment
        return True 
    else:
        return False

def leave_one_out_estimate(X, Y):

    estimate = 0
    
    # K = X.shape[0] # number of folds
    K = 100
    skf = StratifiedKFold(n_splits=K, random_state=0, shuffle=True)
    skf.get_n_splits(X,Y)
    
    c_value = 1.0 # default
    
    for train_index, test_index in skf.split(X,Y):
      x_train, x_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
      
      weight_vector = get_weight_vector(y_train)
      
      # --- Train (training set) ---
      svm_model = classifier_weighted(x_train, y_train, weight_vector, c_value, 'svm')
      
      # --- Test (validation set) ---
      prediction_validation = svm_model.predict(x_test)
          
      # --- Performance ---
      acc = ( np.where((prediction_validation - y_test) == 0)[0].shape[0]) / y_test.shape[0]
        
      if acc < 0.5:
          estimate = estimate + 1

    return estimate
    
#Angle based method
def angle(data_1, target_1, data_2, target_2):
    '''print('\nCalculating angle')
    start_time = time.time()'''
    
    weight_1 = get_weight_vector(target_1)
    weight_2 = get_weight_vector(target_2)

    clf1 = classifier_weighted(data_1, target_1, weight_1, 1.0, 'LogReg')
    clf2 = classifier_weighted(data_2, target_2, weight_2, 1.0, 'LogReg')

    weights1 = clf1.coef_[0]
    weights2 = clf2.coef_[0]

    output = angle_between(weights1, weights2) * 180 / pi

    ''' end_time = time.time()
    run_time = end_time - start_time
    print(f'\nRunning time per combination = {run_time:.2f}')
    print('Angle calculation completed')'''

    return round(output,3)

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    u_vector = vector / np.linalg.norm(vector)
    return u_vector

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle


def grid_search_combination_list(test_idx):
    combination_list = []
    
    for val_idx in range(test_idx-1,0,-1):
        
        for start in range(val_idx):
            
            combination_list.append([[i for i in range(start,val_idx)], [val_idx]])
    
    return combination_list


def train_combination_list(test_idx):
    combination_list = []

    val_idx = test_idx-1
    for start in range(val_idx):
    
        combination_list.append([[i for i in range(start,val_idx+1)], [val_idx]])
    
    return combination_list


# Dynamic Integration
def windowing(datetimes, window_size_minutes):
    """ Returns the indexes of each window for a given seizure with 0% overlap. """
    
    Dict = {}
    window_idx = 0
    begin = 0
    window_idx_list = []

    for i in range(len(datetimes)):
        
        if datetimes[i] > (datetimes[begin] + timedelta(minutes = window_size_minutes)):
            print()
            Dict[window_idx] = np.array(window_idx_list).astype('int')  
            begin = i
            window_idx_list = []
            window_idx = window_idx + 1
            
        else:
            window_idx_list.append(i)
            
    return Dict

def get_ensemble_weights(svm_list, data, target):

    ensemble_scores = []
    
    for svm_model in svm_list:

        prediction = svm_model.predict(data)
        weight_vector = get_weight_vector(target)

        # --- Performance ---
        ss, sp, metric, acc = performance_weighted(target, prediction, weight_vector)

        ensemble_scores.append(acc)

    ensemble_weights = [num/sum(ensemble_scores) for num in ensemble_scores]
    return ensemble_weights



