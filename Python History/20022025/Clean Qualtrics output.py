# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:02:58 2024

@author: benja
"""

import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = 'D:\Surfdrive\Projects\Copy Trading\Results'

# Initialize an empty list to store DataFrames
dataframes_list = []

# Loop through each file in the folder
for idx, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, filename)
        
        # Load the CSV into a DataFrame
        df = pd.read_csv(file_path)
        
        # For all files except the first one, drop the first two columns
        if idx > 0:
           df = df.iloc[2:, :].reset_index(drop=True)
        
        # Drop the specified columns if they exist in the DataFrame
        columns_to_drop = ['botdata', 'dataTraining', 'pricepaths']
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

# Print the first few rows to verify
print(panel_df.head())

# Drop the specified columns
columns_to_drop = ['rounds', 'OngoingReturn', 'ongoingReturn','roundSeries', 'priceSeries', 'assetsSeries']
panel_df = panel_df.drop(columns=[col for col in columns_to_drop if col in panel_df.columns], errors='ignore')

# Convert 'returnall' to integers
def convert_to_int(value):
    try:
        return int(value)  # Try converting directly
    except ValueError:
        # Handle cases where conversion fails
        return 0  # Replace invalid values with 0 (or any default)

if 'returnAll' in panel_df.columns:
    panel_df['returnAll'] = panel_df['returnAll'].apply(convert_to_int)

# Convert 'plotclicks' to strings
def convert_plotclicks(value):
    if isinstance(value, list):
        return str(value)  # Convert lists to their string representation
    elif pd.isna(value):
        return ''  # Replace NaN with an empty string
    else:
        return str(value)  # Convert anything else to string

if 'plotclicks' in panel_df.columns:
    panel_df['plotclicks'] = panel_df['plotclicks'].apply(convert_plotclicks)
    
if 'rankingClicks' in panel_df.columns:
    panel_df['rankingClicks'] = panel_df['rankingClicks'].apply(convert_plotclicks)

# Print the first few rows to verify
print(panel_df.head())

# Export the DataFrame to a CSV file (Stata-compatible format)
panel_df.to_csv("panel_data.csv", index=False)

# Alternatively, save as Stata .dta file
panel_df.to_stata("panel_data.dta", write_index=False)