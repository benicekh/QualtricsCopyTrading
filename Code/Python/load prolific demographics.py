# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:01:39 2025

@author: U338144
"""

import os
import pandas as pd

folder_paths = [
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Results\Pilot\Prolific Demographics",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Results\Pilot\Prolific Demographics"
]

folder_path = next((path for path in folder_paths if os.path.exists(path)), None)

if folder_path is None:
    raise FileNotFoundError("None of the specified folder paths exist.")

print(f"Using folder path: {folder_path}")


dataframes_list = []


for idx, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith('.csv'):  
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        
        # Drop the specified columns if they exist in the DataFrame
        columns_to_drop = ['botdata', 'dataTraining', 'pricepaths']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Append the DataFrame to the list
        dataframes_list.append(df)

# Concatenate all DataFrames into one
final_dataframe = pd.concat(dataframes_list, ignore_index=True)
final_dataframe = final_dataframe[final_dataframe["Status"] == "APPROVED"]

#Convert 'Sex' column: Female -> 1, Male -> 0
final_dataframe["SexCat"] = final_dataframe["Sex"].map({"Female": 1, "Male": 0})
final_dataframe["Country of residenceCat"] = final_dataframe["Country of residence"].map({"United States": 1, "United Kingdom": 0})
final_dataframe["Ethnicity simplifiedCat"] = (final_dataframe["Ethnicity simplified"] == "White").astype(int)
final_dataframe["Age"] = final_dataframe["Age"].astype(int)
final_dataframe["Time taken"] = final_dataframe["Time taken"].astype(int)

averages = final_dataframe.mean(numeric_only=True)
print(averages)