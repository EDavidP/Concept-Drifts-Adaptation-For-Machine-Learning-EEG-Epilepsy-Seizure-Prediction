import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta as t
from sklearn import preprocessing, feature_selection, svm, metrics, model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


def cut_data(datetimes, minutes):
    
    begin_time = datetimes[-1] - t(minutes = minutes) 
    idx = np.where(datetimes>=begin_time)
    
    return idx

def cut_data_hours(hours,datetimes, onset):
    
    begin_time = onset - t(minutes = hours*60)
    idx = np.where(datetimes>=begin_time)
    
    return idx


def construct_target(datetimes_seizure, onset_seizure, SOP, SPH):
    # begin_sph and begin_sop are datetime object in the hours and days format 
    begin_sph = onset_seizure - t(minutes = SPH) 
    begin_sop = begin_sph - t(minutes = SOP) # begin_sop = onset_seizure - t(minutes = SPH) - t(minutes = SOP)
    
    idx_sop = np.where(datetimes_seizure>begin_sop) # idx_sop = np.where((times_seizure>=begin_sop) & (times_seizure<begin_sph))
    idx_sph = np.where(datetimes_seizure>=begin_sph)
    
    target = np.zeros(len(datetimes_seizure), dtype=int)
    target[idx_sop] = 1
    target[idx_sph] = 2
    
    print(f'# Samples preictal: {len(np.where(target==1)[0])}')
    
    return target


def cut_sph(target):
    
    idx_sph = np.where(target==2)
    target = np.delete(target, idx_sph)
    
    return target



def classifier(data, target, c_value):
    
    # Define svm model
    svm_model = svm.LinearSVC(C = c_value, dual = False)
    # Appy fit
    svm_model.fit(data, target)

    # LogisticRegression(random_state=0).fit(data_2, target_2)
    
    return svm_model


def classifier_weighted(data, target, weight, c_value, clf): # Approach B, C, D
    
    if clf == 'svm':
        # Define svm model
        model = svm.LinearSVC(C = c_value, dual = False)
        # Appy fit
        model.fit(data, target, sample_weight = weight)
    elif clf == 'LogReg':
        model = LogisticRegression(random_state=0).fit(data, target, sample_weight = weight)

    return model
    
def class_balancing(target):
    
    # Define majority & minority classes (class with more samples vs. class with less samples)
    idx_class0 = np.where(target==0)[0]
    idx_class1 = np.where(target==1)[0]
    if len(idx_class1)>=len(idx_class0):
        idx_majority_class = idx_class1
        idx_minority_class = idx_class0
    elif len(idx_class1)<len(idx_class0):
        idx_majority_class = idx_class0
        idx_minority_class = idx_class1
    
    # Define number of samples of each group
    n_groups = len(idx_minority_class)
    n_samples = len(idx_majority_class)
    min_samples = n_samples//n_groups
    remaining_samples = n_samples%n_groups
    n_samples_per_group = [min_samples+1]*remaining_samples + [min_samples]*(n_groups-remaining_samples)
    
    # Select one sample from each group of the majority class
    idx_selected = []
    begin_idx = 0
    for i in n_samples_per_group:
        end_idx = begin_idx + i
        
        idx_group = idx_majority_class[begin_idx:end_idx]
        idx = np.random.choice(idx_group)
        idx_selected.append(idx)

        begin_idx = end_idx
        
    # Add samples from the minority class
    [idx_selected.append(idx) for idx in idx_minority_class]

    # Sort selected indexes to keep samples order
    idx_selected = np.sort(idx_selected)
    
    return idx_selected

def get_sop(target):
    
    idx_sop = np.where(target==1)
    
    return idx_sop

def get_last_n_hours(n, datetimes, onset_seizure, SPH):

    last_n_hours_begin = onset_seizure - t(minutes = SPH) - t(minutes = int(n*60))
    idx_last_n_hours = np.where(datetimes>=last_n_hours_begin)

    return idx_last_n_hours

def get_second_last_hour(datetimes, onset_seizure, SPH):

    begin = onset_seizure - t(minutes = SPH) - t(minutes = int(2*60))
    end = onset_seizure - t(minutes = SPH) - t(minutes = int(1*60))
    idx = np.where((datetimes>=begin) & (datetimes<=end))

    return idx
    
def get_weight_vector(target):
    
    idx_class0 = np.where(target==0)[0]
    idx_class1 = np.where(target==1)[0]
    
    class0_weight = len(target) / len(idx_class0)  
    class1_weight = len(target) / len(idx_class1)
    
    weight_vector = np.zeros(len(target))

    weight_vector[idx_class0] = class0_weight
    weight_vector[idx_class1] = class1_weight
    
    return weight_vector

def find_redundant_features(data):
    redundant_features_index=[]
    # Apply pearson correlation to find features with corr>0.95
    for i in range(0,data.shape[1]):
        for j in range(i,data.shape[1]):
            if i!=j and abs(np.corrcoef(data[:,i],data[:,j])[0][1])>0.95:
                if j not in redundant_features_index:
                    redundant_features_index.append(j)
                
    return redundant_features_index

def standardization(data):
    
    # Define scaler
    scaler = preprocessing.StandardScaler()
    # Apply fit
    scaler.fit(data)
    
    return scaler


def get_selected_features(selector): # ANOVA or Random Forest
    
    #return a mask of the features selected
    mask=selector.get_support(indices=True)
    
    selected_features_index=mask
    
    return selected_features_index


def performance(target, prediction):
    
    tn, fp, fn, tp = metrics.confusion_matrix(target, prediction).ravel()
    sensitivity = tp/(tp+fn)  
    specificity = tn/(tn+fp)
    metric = np.sqrt(sensitivity * specificity)
    accuracy = (tp+tn)/(tp+fn+tn+fp)  
   
    return sensitivity, specificity, metric, accuracy


def performance_weighted(target, prediction, weight): # Approach B, C, D
    
    tn, fp, fn, tp = metrics.confusion_matrix(target, prediction, sample_weight = weight).ravel()
    sensitivity = tp/(tp+fn)  
    specificity = tn/(tn+fp)
    metric = np.sqrt(sensitivity * specificity)
    accuracy = (tp+tn)/(tp+fn+tn+fp)  
   
    return sensitivity, specificity, metric, accuracy

def remove_sop(target):
    
    idx_sop = np.where(target==1)
    target = np.delete(target, idx_sop)
    
    return target

def remove_redundant_features(data, redundant_features_index):
    # deleting redundant features   
    return np.delete(data, redundant_features_index, axis=1)

def select_features(data, target, n_features, algorithm):  
    
    if algorithm == 'ANOVA':
    
        # Define feature selection using ANOVA
        selector = feature_selection.SelectKBest(score_func = feature_selection.f_classif, k = n_features)  
        # Apply feature selection
        selector.fit(data, target)
    
    elif algorithm == 'Random Forest Ensemble':
        # Define feature selection using Random Forest Ensemble
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', class_weight='balanced')
        # Train the classifier
        clf.fit(data, target)
        
        selector = SelectFromModel(clf, max_features = n_features,  threshold=-np.inf)
        # Train the selector
        selector.fit(data, target)
        
    return selector