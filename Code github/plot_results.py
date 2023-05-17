import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import math

from auxiliary_func import get_selected_features

def fig_performance(patient, info_train, info_test, approach, split_type):
    
    # Information
    ss = [info[-6] for info in info_test]
    fprh = [info[-5] for info in info_test]
    labels = [info[0] for info in info_train]
        
    # Figure
    fig = plt.figure(figsize=(20, 10))
    
    x = np.arange(len(ss))
    width = 0.3
    
    # Sensitivity
    ss_bar = plt.bar(x-width/2, ss, width, color='yellowgreen', label = 'SS')
    plt.bar_label(ss_bar, fmt='%.2f', padding=3)
    plt.xlabel('SOP')
    plt.ylabel('SS', color='yellowgreen', fontweight='bold')
    plt.ylim([0, 1.05])
    
    # FPR/h
    plt.twinx() # second axis
    fprh_bar = plt.bar(x+width/2, fprh, width, color='sandybrown', label = 'FPR/h')
    plt.bar_label(fprh_bar, fmt='%.2f', padding=3)
    plt.ylabel('FPR/h', color='sandybrown', fontweight='bold')
    
    plt.xticks(x, labels=labels)
    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.annotate('* final result', xy=(1,0), xycoords='axes fraction', xytext=(-60,-50), textcoords='offset points')

    plt.title(f'Performance (patient {patient})')

    plt.savefig(f'Results/{approach}/{split_type}/Performance (patient {patient})')
    plt.close()    


def fig_final_performance(final_information, approach, split_type):
    
    # Information
    ss = [info[-6] for info in final_information]
    fprh = [info[-5] for info in final_information]
    sop = [info[4] for info in final_information]
    
    labels = [f'patient {info[0]}' for info in final_information]
    p_values = [info[-1] for info in final_information]
    labels = [f'*{labels[info]}'  if p_values[info]<0.05 else f'{labels[info]}' for info in range(len(final_information))] # validated patients

    # Figure
    fig = plt.figure(figsize=(20, 10))
    
    x = np.arange(len(final_information))
    width = 0.3
    spare_width = 0.5
    
    # Sensitivity
    ss_bar = plt.bar(x-width/2, ss, width, color='yellowgreen', label = 'SS')
    plt.bar_label(ss_bar, fmt='%.2f', padding=3)
    plt.ylabel('SS', color='yellowgreen', fontweight='bold')
    plt.ylim([0, 1.05])
    plt.xticks(x, labels=labels, rotation=90)
    plt.xlim(x[0]-spare_width,x[-1]+spare_width)

    # FPR/h
    plt.twinx() # second axis
    fprh_bar = plt.bar(x+width/2, fprh, width, color='sandybrown', label = 'FPR/h')
    plt.bar_label(fprh_bar, fmt='%.2f', padding=3)
    plt.ylabel('FPR/h', color='sandybrown', fontweight='bold')
    
    plt.table(cellText=[sop], rowLabels=['SOP'], cellLoc='center', bbox=[0, -0.25, 1, 0.05], edges='horizontal') # BBOX: [shift on x-axis, gap between plot & table, width, height]
    plt.subplots_adjust(bottom=0.25)
    
    ss_final = [np.mean(ss), np.std(ss)]
    fprh_final = [np.mean(fprh), np.std(fprh)]    
    print(f'\n\n--- FINAL TEST PERFORMANCE (selected SOPs - mean) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (mean) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 1, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)

    ss_final = [np.median(ss), np.percentile(ss, 75) - np.percentile(ss, 25)]
    fprh_final = [np.median(fprh), np.percentile(fprh, 75) - np.percentile(fprh, 25)]
    print(f'\n\n--- FINAL TEST PERFORMANCE (selected SOPs - median) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (median) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 0.88, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)

    plt.annotate('* statistically validated', xy=(1,0), xycoords='axes fraction', xytext=(-115,-130), textcoords='offset points')

    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.title('Performance per patient')
    
    plt.savefig(f'Results/{approach}/{split_type}/Performance')
    plt.close()
        
    
def fig_performance_per_patient(information_test, approach, split_type):
    
    # Information
    patients = information_test.keys()
    ss_mean = []
    ss_std = []
    fprh_mean = []
    fprh_std = []
    for patient in patients:
        ss = [info[5] for info in information_test[patient]]
        fprh = [info[6] for info in information_test[patient]]
        
        ss_mean.append(np.mean(ss))
        ss_std.append(np.std(ss))
        fprh_mean.append(np.mean(fprh))
        fprh_std.append(np.std(fprh))
    labels = patients #labels = [f'patient {patient}' for patient in patients]

    # Figure
    fig = plt.figure(figsize=(20, 10))
    
    x = np.arange(len(patients))
    width = 0.3
    
    # Sensitivity
    plt.bar(x-width/2, ss_mean, width, yerr=ss_std, color='yellowgreen', label = 'SS', error_kw=dict(elinewidth=0.5, capsize=5))
    plt.ylabel('SS', color='yellowgreen', fontweight='bold')
    plt.ylim([0, 1.05])
    plt.xlabel('Patient')
    plt.xticks(x, labels=labels)
    [plt.annotate(str(round(ss_mean[i],2)),(x[i]-width,ss_mean[i]+plt.ylim()[1]/100),fontsize=7.5) for i in range(len(ss_mean))]

    # FPR/h
    plt.twinx() # second axis
    plt.bar(x+width/2, fprh_mean, width, yerr=fprh_std, color='sandybrown', label = 'FPR/h', error_kw=dict(elinewidth= 0.5, capsize=5))
    plt.ylabel('FPR/h', color='sandybrown', fontweight='bold')
    plt.ylim(bottom=0)
    [plt.annotate(str(round(fprh_mean[i],2)),(x[i],fprh_mean[i]+plt.ylim()[1]/100),fontsize=7.5) for i in range(len(fprh_mean))]
 
    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.title('Performance per patient')    

    ss_final = [np.mean(ss_mean), np.std(ss_mean)]
    fprh_final = [np.mean(fprh_mean), np.std(fprh_mean)]    
    print(f'\n\n--- FINAL TEST PERFORMANCE (all SOPs - mean) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 1, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)
    
    ss_final = [np.median(ss_mean), np.percentile(ss_mean, 75) - np.percentile(ss_mean, 25)]
    fprh_final = [np.median(fprh_mean), np.percentile(fprh_mean, 75) - np.percentile(fprh_mean, 25)]
    print(f'\n\n--- FINAL TEST PERFORMANCE (all SOPs - median) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (mean) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 0.9, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)
    
    plt.savefig(f'Results/{approach}/{split_type}/Performance per patient')
    plt.close()

    

def fig_parameters_selection(final_information, approach, split_type, channels, features):
    # sop, features, C, Channels
    c_pot_list = np.arange(start = -10, stop = 10, step = 2, dtype=float) 
    C = 2**c_pot_list # parameter of SVM classifier
    SOP_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]

    patients = [info[0] for info in final_information]

    idx_final_selected_features_dict = {0:[],1:[],2:[],3:[],4:[]}
    idx_final_selected_sop_dict = {0:[],1:[],2:[],3:[],4:[]}
    idx_final_selected_cost_dict = {0:[],1:[],2:[],3:[],4:[]}
    for i in range(len(patients)):
        model = pickle.load(open(f'Models/{approach}/{split_type}/model_patient{patients[i]}_{split_type}','rb'))
        n_iterations = len(model)-1
        final_SOP_list = [model[iteration]['iteration_info_train'][0] for iteration in range(n_iterations)]
        final_cost_list = [int(math.log(model[iteration]['iteration_info_train'][2],2)) for iteration in range(n_iterations)] # potências

        idx_final_selected_features_patient_dict = {0:[],1:[],2:[],3:[],4:[]}
        idx_final_selected_sop_patient_dict = {0:[],1:[],2:[],3:[],4:[]}
        idx_final_selected_cost_patient_dict = {0:[],1:[],2:[],3:[],4:[]}

        for iteration in range(n_iterations):
            final_selector = model[iteration][final_SOP_list[iteration]]['selector']

            idx_final_selected_features = np.concatenate([get_selected_features(final_selector[idx]) for idx in range(len(final_selector))]) # concatenate features from 31 models of best SOP
            idx_final_selected_features_dict[iteration].append(idx_final_selected_features)
            idx_final_selected_features_patient_dict[iteration].append(idx_final_selected_features)

            idx_final_selected_sop = final_SOP_list[iteration]
            idx_final_selected_sop_dict[iteration].append([idx_final_selected_sop])
            idx_final_selected_sop_patient_dict[iteration].append([idx_final_selected_sop])

            idx_final_selected_cost = final_cost_list[iteration]
            idx_final_selected_cost_dict[iteration].append([idx_final_selected_cost])
            idx_final_selected_cost_patient_dict[iteration].append([idx_final_selected_cost])

        # Figure: final selected features (patient)
        idx_final_selected_features_list=[]
        idx_final_selected_sop_list=[]
        idx_final_selected_cost_list=[]
        for iteration in range(n_iterations):
            idx_final_selected_features_list.append(np.concatenate(idx_final_selected_features_patient_dict[iteration]))
            idx_final_selected_sop_list.append(np.concatenate(idx_final_selected_sop_patient_dict[iteration]))
            idx_final_selected_cost_list.append(np.concatenate(idx_final_selected_cost_patient_dict[iteration]))
        
        if patients[i] == '30802' or patients[i] == '55202' or patients[i] == '114702' or patients[i] == '59102' or patients[i] == '75202':
            fig_ps(n_iterations,idx_final_selected_features_list, idx_final_selected_cost_list, idx_final_selected_sop_list, f'{patients[i]}_final', approach, split_type, features, channels, SOP_list, c_pot_list)  

    # Figure: final selected features (all patients)
    idx_final_selected_features_list=[]
    idx_final_selected_sop_list=[]
    idx_final_selected_cost_list=[]
    max_n_iterations = 5
    for iteration in range(max_n_iterations):
        idx_final_selected_features_list.append(np.concatenate(idx_final_selected_features_dict[iteration]))
        idx_final_selected_sop_list.append(np.concatenate(idx_final_selected_sop_dict[iteration]))
        idx_final_selected_cost_list.append(np.concatenate(idx_final_selected_cost_dict[iteration]))
    
    fig_ps(max_n_iterations,idx_final_selected_features_list, idx_final_selected_cost_list, idx_final_selected_sop_list, 'final', approach, split_type, features, channels, SOP_list, c_pot_list)  
    

def fig_ps(n_iterations, idx_selected_features, idx_final_selected_cost, idx_final_selected_sop, fig_name, approach, split_type, features, channels, SOP_list, c_pot_list):  
    
    iteration_numbers = ['#4','#5','#6','#7','#8']
    features_freq_all_iter_matrix = np.zeros((5,len(features)))
    channels_freq_all_iter_matrix = np.zeros((5,len(channels)))
    cost_freq_all_iter_matrix = np.zeros((5,len(c_pot_list)))
    sop_freq_all_iter_matrix = np.zeros((5,len(SOP_list)))
    for iteration in range(n_iterations):

        features_freq = {feature:0 for feature in features}
        channels_freq = {channel:0 for channel in channels}
        features_channels_freq = {feature+'_'+channel:0 for feature in features for channel in channels}
        for i in idx_selected_features[iteration]:
            
            idx_feature = i//len(channels)
            feature_name = features[idx_feature]
            features_freq[feature_name] += 1 
            
            idx_channel = i%len(channels)
            channel_name = channels[idx_channel]
            channels_freq[channel_name] += 1
            
            features_channels_freq[feature_name+'_'+channel_name] += 1

        cost_freq = {int(cost):0 for cost in c_pot_list}
        for i in idx_final_selected_cost[iteration]:

            cost_name = i
            cost_freq[cost_name] += 1 

        sop_freq = {SOP:0 for SOP in SOP_list}
        for i in idx_final_selected_sop[iteration]:

            SOP_name = i
            sop_freq[SOP_name] += 1 
            
       
        # Translate from number of occurrences to relative frequency
        features_freq = {name: n_occur/len(idx_selected_features[iteration]) for name,n_occur in features_freq.items()}
        channels_freq = {name: n_occur/len(idx_selected_features[iteration]) for name,n_occur in channels_freq.items()}
        features_channels_freq = {name: n_occur/len(idx_selected_features[iteration]) for name,n_occur in features_channels_freq.items()}
        cost_freq = {name: n_occur/len(idx_final_selected_cost[iteration]) for name,n_occur in cost_freq.items()}
        sop_freq = {name: n_occur/len(idx_final_selected_sop[iteration]) for name,n_occur in sop_freq.items()}

        features_freq_all_iter_matrix[iteration,:] = list(features_freq.values())
        channels_freq_all_iter_matrix[iteration,:] = list(channels_freq.values())
        cost_freq_all_iter_matrix[iteration,:] = list(cost_freq.values())
        sop_freq_all_iter_matrix[iteration,:] = list(sop_freq.values())

    mapp= [features_freq_all_iter_matrix[:,0:19], features_freq_all_iter_matrix[:,19:42], features_freq_all_iter_matrix[:,42:59]]
    features_mapp=[features[0:19],features[19:42],features[42:59]]
    
    if fig_name == 'final':
        norm1 = matplotlib.colors.Normalize(vmin=0, vmax=0.5)
    else:
        norm1 = matplotlib.colors.Normalize(vmin=0, vmax=0.5)
    norm2 = matplotlib.colors.Normalize(vmin=0, vmax=1)

    fig = plt.figure(figsize=(14, 25))
    gs = gridspec.GridSpec(5, 2)
    ax1 = plt.subplot(gs[0,0])
    if fig_name == 'final':
        ax1.set_title('Relative frequency of the selected SOPs')
        im1 = ax1.imshow(sop_freq_all_iter_matrix, cmap='bone_r', norm=norm2)
        ax1.set_xticks(np.arange(sop_freq_all_iter_matrix.shape[1]), labels=SOP_list)
        ax1.set_yticks(np.arange(sop_freq_all_iter_matrix.shape[0]), labels=iteration_numbers)
        ax1.set_ylabel('Seizure')
    else:
        ax1.set_title('Selected SOPs')
        im1 = ax1.imshow(sop_freq_all_iter_matrix, cmap='bone_r', norm=norm2)
        ax1.set_xticks(np.arange(sop_freq_all_iter_matrix.shape[1]), labels=SOP_list)
        ax1.set_yticks(np.arange(sop_freq_all_iter_matrix.shape[0]), labels=iteration_numbers)
        ax1.set_ylabel('Seizure')

    ax2 = plt.subplot(gs[0,1])
    if fig_name == 'final':
        ax2.set_title('Relative frequency of the selected SVM costs')
        im2 = ax2.imshow(cost_freq_all_iter_matrix, cmap='bone_r', norm=norm2)
        c_pot_list_labels = []
        for c_pot in c_pot_list:
            c_pot_list_labels.append('2^'+str(int(c_pot)))
        ax2.set_xticks(np.arange(cost_freq_all_iter_matrix.shape[1]), labels=c_pot_list_labels)
        ax2.set_yticks(np.arange(cost_freq_all_iter_matrix.shape[0]), labels=iteration_numbers)
        ax2.set_ylabel('Seizure')
    else:
        ax2.set_title('Selected SVM costs')
        im2 = ax2.imshow(cost_freq_all_iter_matrix, cmap='bone_r', norm=norm2)
        c_pot_list_labels = []
        for c_pot in c_pot_list:
            c_pot_list_labels.append('2^'+str(int(c_pot)))
        ax2.set_xticks(np.arange(cost_freq_all_iter_matrix.shape[1]), labels=c_pot_list_labels)
        ax2.set_yticks(np.arange(cost_freq_all_iter_matrix.shape[0]), labels=iteration_numbers)
        ax2.set_ylabel('Seizure')

    ax3 = plt.subplot(gs[1,0:2])
    ax3.set_title('Relative frequency of the selected channels')
    im3 = ax3.imshow(channels_freq_all_iter_matrix, cmap='bone_r', norm=norm1)
    ax3.set_xticks(np.arange(channels_freq_all_iter_matrix.shape[1]), labels=channels)
    ax3.set_yticks(np.arange(channels_freq_all_iter_matrix.shape[0]), labels=iteration_numbers)
    ax3.set_ylabel('Seizure')

    ax4 = plt.subplot(gs[2,0:2])
    ax4.set_title('Relative frequency of the selected features')
    im4 = ax4.imshow(mapp[0], cmap='bone_r', norm=norm1)
    ax4.set_xticks(np.arange(mapp[0].shape[1]), labels=features_mapp[0], rotation = 45, ha="right", rotation_mode="anchor")
    ax4.set_yticks(np.arange(mapp[0].shape[0]), labels=iteration_numbers)
    ax4.set_ylabel('Seizure')

    ax5 = plt.subplot(gs[3,0:2])
    im5 = ax5.imshow(mapp[1], cmap='bone_r', norm=norm1)
    ax5.set_xticks(np.arange(mapp[1].shape[1]), labels=features_mapp[1], rotation = 45, ha="right", rotation_mode="anchor")
    ax5.set_yticks(np.arange(mapp[1].shape[0]), labels=iteration_numbers)
    ax5.set_ylabel('Seizure')

    ax6 = plt.subplot(gs[4,0:2])
    im6 = ax6.imshow(mapp[2], cmap='bone_r', norm=norm1)
    ax6.set_xticks(np.arange(mapp[2].shape[1]), labels=features_mapp[2], rotation = 45, ha="right", rotation_mode="anchor")
    ax6.set_yticks(np.arange(mapp[2].shape[0]), labels=iteration_numbers)
    ax6.set_ylabel('Seizure')

    ims = [im1, im2, im3, im4, im5, im6]
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    if fig_name == 'final':
        plt.colorbar(ims[0], ax=axes[0:2], label='Relative frequency', aspect=4.5)
    else:
        cbar = plt.colorbar(ims[0], ax=axes[0:2], ticks=[0, 1], aspect=4.5)
        cbar.set_ticklabels(['Not selected', 'Selected'])
    plt.colorbar(ims[2], ax=axes[2:6], label='Relative frequency')
    plt.savefig(f'Results/{approach}/{split_type}/colormaps/feature_channel_selection_final_sop_cost{fig_name}', bbox_inches='tight')
    plt.close()



def fig_test(patient, preictal, target, prediction, firing_power, threshold, alarm, approach, split_type):
    
    fig, ax = plt.subplots(4, 1, figsize=(20, 10))
    
    # Target
    ax[0].plot(target, 'o', markersize=2, color='orange')
    ax[0].set_xlim([0, len(target)])
    ax[0].set_ylabel('Target', color='orange')
    ax[0].set_yticks([0,1])
    ax[0].set_yticklabels(['Interictal','Preictal'])

    # Prediction
    ax[1].plot(prediction, 'o', markersize=2)
    ax[1].set_xlim([0, len(prediction)])
    ax[1].set_ylabel('SVM Prediction', color='blue')
    ax[1].set_yticks([0,1])
    ax[1].set_yticklabels(['Interictal','Preictal'])
    
    # Firing power
    ax[2].plot(firing_power, color='red')
    ax[2].plot(np.full(len(firing_power),threshold), color='green', label='Threshold')
    ax[2].set_xlim([0, len(firing_power)])
    ax[2].set_ylabel('Firing Power', color='red')
    ax[2].set_ylim([-0.05, 1.05]) 
    ax[2].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[2].grid(axis='y')
    ax[2].legend(loc='upper right')
    
    # Alarms
    ax[3].plot(alarm, color='yellow')
    ax[3].set_xlim([0, len(alarm)])
    ax[3].set_ylabel('Alarms', color='yellow')
    ax[3].set_yticks([0,1])
    ax[3].set_yticklabels(['Interictal','Preictal'])   

    ax[0].set_title(f'Test: patient {patient} | preictal={preictal}')
    ax[3].set_xlabel('Samples')
    fig.align_labels()

    fig.savefig(f'Results/{approach}/{split_type}/test_pat{patient}_preictal{preictal}')
    plt.close(fig)