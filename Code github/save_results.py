import numpy as np
import xlsxwriter as xw


def save_results(information_general, information_train, information_test, approach, split_type):
    
    # Create xlsx
    path = f'Results/{approach}/{split_type}/Results.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})

    patients = list(information_general.keys())
    for patient in patients:
        info_general = information_general[patient]
        info_train = information_train[patient]
        info_test = information_test[patient]
        
        # Create sheet
        ws = wb.add_worksheet(f'pat_{patient}')

        # Header format
        format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
        format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
        format_test = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})
        
        # Insert Header
        header_general = ['Patient','#Seizures train','#Seizures test','SPH','SOP']
        header_train = ['#Features','Cost','SS samples','SP samples','Metric']
        header_test = ['SS samples','SP samples','Threshold','#Predicted','#False Alarms','SS','FPR/h','SS surrogate mean','SS surrogate std','tt','p-value']
        
        row = 0
        col = 0
        ws.write_row(row, col, header_general, format_general)
        col = len(header_general)
        ws.write_row(row, col, header_train, format_train)
        col = col + len(header_train)
        ws.write_row(row, col, header_test, format_test)
    
        # Insert data
        row = 1
        col = 0
        ws.write_row(row, col, info_general)
        
        info = [info_train[i]+info_test[i] for i in range(len(info_train))]
        col = len(info_general)
        for i in info:
            ws.write_row(row, col, i)
            row += 1

    wb.close()

def save_final_results(final_information, approach, split_type):
    
    # Create xlsx
    path = f'Results/{approach}/{split_type}/Final_results.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})
    ws = wb.add_worksheet('Final results')
    
    # Header format
    format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
    format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
    format_test = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})
    
    # Insert Header
    header_general = ['Patient','#Seizures train','#Seizures test','SPH','SOP']
    header_train = ['#Features','Cost','SS samples','SP samples','Metric']
    header_test = ['SS samples','SP samples','Threshold','#Predicted','#False Alarms','SS','FPR/h','SS surrogate mean','SS surrogate std','tt','p-value']
            
    row = 0
    col = 0
    ws.write_row(row, col, header_general, format_general)
    col = len(header_general)
    ws.write_row(row, col, header_train, format_train)
    col = col + len(header_train)
    ws.write_row(row, col, header_test, format_test)
    
    # Insert data
    row = 1
    col = 0
    for i in final_information:
        ws.write_row(row, col, i)
        row += 1

    wb.close()


def select_final_result(info_train, approach):

    metrics = [info[5] for info in info_train] # metric (Classifier)
    idx_best = np.argmax(metrics)
 
    return idx_best

