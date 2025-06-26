# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:52:05 2024

@author: benja
"""
import math 

data_copier = {}
for j in range (2):
    data_copier[j] = {}
    for i in range (20):
        data_copier[j][i+1] = {}
        for stage in range(10):
            stage_key = f"stage_{stage}"
            data_copier[j][i+1][stage_key] = {
                "CRRA_-1.5": 0,
                "CRRA_0": 0,
                "CRRA_1": 0,
                "CRRA_3": 0,
                "CRRA_6": 0
            }
    
data_rank = {}
for j in range (2):
    data_rank[j] = {}
    for i in range (20):
        data_rank[j][i+1] = {}
        for stage in range(10):
            stage_key = f"stage_{stage}"
            data_rank[j][i+1][stage_key] = {
                "rank_1": 0,
                "rank_2": 0,
                "rank_3": 0,
                "rank_4": 0,
                "rank_5": 0
            }


# Iterate over rows
for _, row in panel_df.iterrows():
    if row['followersTreatment'] == 1:
        continue  # Skip row if followersTreatment is 1

    if row['phaseName'] == 'training':
        continue  # Skip row if phaseName is "training"

    if row['nameTreatment'] == 1:
        ps = row['pathseries']
        if math.isnan(ps):
            continue
        ps = int(row['pathseries'])
        st = row['phase']
        tl = row['tradeLeader']
        ra = row['TLrank']
        phase_key = f"stage_{st}"
        
        
        data_rank[1][ps][phase_key][ra] += 1
        data_copier[1][ps][phase_key][tl] += 1
    else:
        ps = row['pathseries']
        if math.isnan(ps):
            continue
        ps = int(row['pathseries'])
        st = row['phase']
        tl = row['tradeLeader']
        ra = row['TLrank']
        phase_key = f"stage_{st}"
        
        
        data_rank[0][ps][phase_key][ra] += 1
        data_copier[0][ps][phase_key][tl] += 1
        
data_rank_shares = {}

for key1 in data_rank:
    data_rank_shares[key1] = {}
    for key2 in data_rank[key1]:
        data_rank_shares[key1][key2] = {}
        for stage in data_rank[key1][key2]:
            stage_data = data_rank[key1][key2][stage]
            total = sum(stage_data.values())  # Compute sum of values
            if total > 0:  # Avoid division by zero
                data_rank_shares[key1][key2][stage] = {k: v / total for k, v in stage_data.items()}
            else:
                data_rank_shares[key1][key2][stage] = {k: 0 for k in stage_data}  # Handle zero total case

        
folder_path = r"D:\Surfdrive\Projects\Copy Trading\Trading game"
for r in range (20):
    string = "const TLs = {"
    string += "\n"    
    for j in range (2):
        string += "\t" 
        treat = str(j)
        string += f'{treat}: '
        string += '{'
        string += "\n"
        for i in range(1, len(data_rank_shares[j][r+1])+1) :
            stage = "stage_" + str(i)
            stagestring = "stage_" + str(i-1)
            string += "\t\t"
            string += f'{stagestring}: '
            string += '{'
            string += "\n"
            for key in data_rank_shares[j][r+1][stagestring].keys():
                string += "\t\t\t"
                string += f'"{key}": '
                #string += str(f'{(data_rank[j][r+1][stagestring][key])/total},')
                string += str(f'{data_rank_shares[j][r+1][stagestring][key]},')
                string += "\n"
            string += "\t\t\t"
            string += "},"
            string += "\n"
        string += "\t\t"
        string += "},"
        string += "\n"
    string += "};"
               
    print(string)
    
    def export_string_to_txt_file(string, file_name):
        with open(file_name, "w") as file:
            file.write(string)
    
    def replace_nan(input_string):
        return input_string.replace('nan', 'NaN')
    
    string = replace_nan(string)
    file_name = f"javastrings_TLs_series_{r+1}.txt"
    subfolder = 'TLdata'
    file_path = os.path.join(subfolder, file_name)
    file_path = os.path.join(folder_path, file_path)
    export_string_to_txt_file(string, file_path)