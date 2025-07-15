# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:52:05 2024

@author: benja
"""
import math 
import os
import pandas as pd


### SET PATHS ###
results_paths = [
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Results\Pilot 2",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Results\Pilot 2"
]
base_path = next((p for p in results_paths if os.path.exists(p)), None)
if base_path is None:
    raise FileNotFoundError("No valid results folder found.")

cleaned_path = os.path.join(base_path, "Cleaned")

# Initialize an empty list to store DataFrames
dataframes_list = []

# Loop through each file in the folder
for idx, filename in enumerate(os.listdir(base_path)):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        data_file_path = os.path.join(base_path, filename)
        
        # Load the CSV into a DataFrame
        df = pd.read_csv(data_file_path)
        
        # For all files except the first one, drop the first two columns
        if idx > 0:
           df = df.iloc[2:, :].reset_index(drop=True)
        
        # Drop the specified columns if they exist in the DataFrame
        columns_to_drop = ['botdata', 'dataTraining', 'pricepaths' , 'DataElicitation', 'TLs']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Append the DataFrame to the list
        dataframes_list.append(df)

# Concatenate all DataFrames into one
final_dataframe = pd.concat(dataframes_list, ignore_index=True)

# Print the final DataFrame's shape to confirm
print(f"Final DataFrame shape: {final_dataframe.shape}")

import json

# Drop rows where the 'Data' column is NaN
final_dataframe = final_dataframe.dropna(subset=['Data']).reset_index(drop=True)

# Initialize an empty dictionary to store the parsed JSON data
json_data_dict = {}

# Iterate through the rows of the DataFrame
for index, row in final_dataframe.iterrows():
    try:
        # Parse the JSON string in the 'Data' column
        parsed_json = json.loads(row['Data'])
        
        # Merge the JSON data with the other columns in the row
        combined_data = {**parsed_json}  # Start with the parsed JSON data
        for col in final_dataframe.columns:
            if col != 'Data':  # Avoid duplicating the 'Data' column
                combined_data[col] = row[col]
        
        # Add the combined data to the dictionary with the row number as key
        json_data_dict[index] = combined_data
    except json.JSONDecodeError as e:
        # Handle cases where the JSON data is malformed
        print(f"Row {index} contains invalid JSON data: {row['Data']}")
        json_data_dict[index] = None  # Optional: Assign None for invalid JSON rows

# Print a sample of the resulting dictionary
print("Sample of the JSON dictionary:", {k: json_data_dict[k] for k in list(json_data_dict.keys())[:5]})

# Initialize a list to store rows for the new DataFrame
rows = []

# Iterate through the main dictionary
for main_key, main_data in json_data_dict.items():
    if main_data is None:  # Skip if the main data is None
        continue
    
    # Extract the stagesummaries, which might be a list
    stagesummaries = main_data.get('stagesummaries', [])
    
    # Ensure stagesummaries is iterable (handle list or dict cases)
    if isinstance(stagesummaries, list):
        # Iterate through the list and treat each item as a stage_data dictionary
        for stage_idx, stage_data in enumerate(stagesummaries):
            # Flatten the stage_data dictionary into a single row
            row = {'stage_key': stage_idx}  # Use the index of the list as the key
            
            # Add all values from stage_data
            if isinstance(stage_data, dict):
                row.update(stage_data)
            
            # Add all other key-value pairs from main_data (excluding stagesummaries)
            for key, value in main_data.items():
                if key != 'stagesummaries':  # Exclude the nested list
                    row[key] = value
            
            # Add the completed row to the list
            rows.append(row)
    elif isinstance(stagesummaries, dict):
        # Handle stagesummaries as a dictionary
        for stage_key, stage_data in stagesummaries.items():
            # Flatten the stage_data dictionary into a single row
            row = {'stage_key': stage_key}
            
            # Add all values from stage_data
            if isinstance(stage_data, dict):
                row.update(stage_data)
            
            # Add all other key-value pairs from main_data (excluding stagesummaries)
            for key, value in main_data.items():
                if key != 'stagesummaries':  # Exclude the nested dictionary
                    row[key] = value
            
            # Add the completed row to the list
            rows.append(row)

# Create a DataFrame from the list of rows
panel_df = pd.DataFrame(rows)


data_copier = {}
for j in range (2):
    data_copier[j] = {}
    for i in range (8):
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
    for i in range (8):
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
        
    if row['path'] == 'training':
        continue  # Skip row if path is "training"

    if row['nameTreatment'] == "1":
        row['pathoption'] = int(row['pathoption'])
        ps = row['pathoption']
        if math.isnan(ps):
            continue
        ps = int(row['pathoption'])
        st = row['phase']
        tl = row['tradeLeader']
        ra = row['TLrank']
        phase_key = f"stage_{st}"
        
        print(ps)
        print(phase_key)
        print(ra)
        data_rank[1][ps][phase_key][ra] += 1
        data_copier[1][ps][phase_key][tl] += 1
    else:
        row['pathoption'] = int(row['pathoption'])
        ps = row['pathoption']
        if math.isnan(ps):
            continue
        ps = int(row['pathoption'])
        st = row['phase']
        tl = row['tradeLeader']
        ra = row['TLrank']
        phase_key = f"stage_{st}"
        
        print(ps)
        print(phase_key)
        print(ra)
        print(row)
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
                data_rank_shares[key1][key2][stage] = {k: round((v / total) * 100, 2) for k, v in stage_data.items()}
            else:
                data_rank_shares[key1][key2][stage] = {k: 0 for k in stage_data}  # Handle zero total case

# Define suffixes list (matching the JS order; Python index 1 corresponds to JS index 1)
suffixes = [
    None,
    'no_risk_start_bottom_high',
    'no_risk_start_bottom_low',
    'no_risk_start_top_high',
    'no_risk_start_top_low',
    'risk_start_bottom_high',
    'risk_start_bottom_low',
    'risk_start_top_high',
    'risk_start_top_low'
]

folder_path = r"D:\Surfdrive\Projects\Copy Trading\Trading game"

def export_string_to_txt_file(string, file_name):
    with open(file_name, "w") as file:
        file.write(string)

def replace_nan(input_string):
    return input_string.replace('nan', 'NaN')

for r in range(1, 9):  # r from 1 to 8 inclusive to match suffixes[1]–[8]
    string = "window.TLs = {}"
    string += "\n"
    string += "TLs = {\n"
    for j in range(2):
        string += "\t" 
        treat = str(j)
        string += f'{treat}: {{\n'
        for i in range(1, len(data_rank_shares[j][r]) + 1):
            stage = f"stage_{i}"
            stagestring = f"stage_{i - 1}"
            string += f"\t\t{stagestring}: {{\n"
            for key in data_rank_shares[j][r][stagestring]:
                string += f'\t\t\t"{key}": {data_rank_shares[j][r][stagestring][key]},\n'
            string += "\t\t},\n"
        string += "\t},\n"
    string += "};"

    string = replace_nan(string)
    suffix = suffixes[r]
    file_name = f"javastrings_TLs_series_{suffix}.txt"
    subfolder = 'TLdata'
    file_path = os.path.join(folder_path, subfolder, file_name)
    export_string_to_txt_file(string, file_path)
    
    file_name = f"javastrings_TLs_series_{suffix}.js"
    subfolder = 'TLdata'
    file_path = os.path.join(folder_path, subfolder, file_name)
    export_string_to_txt_file(string, file_path)