# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:32:12 2025

@author: benja
"""

import os
import pandas as pd
import seaborn as sns
import re
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import minimize
from math import floor

### SET PATHS ###
results_paths = [
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Data\CRRA Test",

]
base_path = next((p for p in results_paths if os.path.exists(p)), None)
if base_path is None:
    raise FileNotFoundError("No valid results folder found.")
graphs_path = os.path.join(base_path, "Graphs")
demo_path = os.path.join(base_path, "Prolific Demographics")
cleaned_path = os.path.join(base_path, "Cleaned")
os.makedirs(graphs_path, exist_ok=True)
os.makedirs(graphs_path, exist_ok=True)

folder_paths = [
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Data\CRRA Test",
]

bot_path = next((path for path in folder_paths if os.path.exists(path)), None)

if bot_path is None:
    raise FileNotFoundError("None of the specified folder paths exist.")

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
    stagesummaries = main_data.get('rounds', [])
    
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

panel_df_clean = panel_df[panel_df['phase'] != 'training']

columns_to_keep = ['ResponseId', 'mleR', 'mleLambda', 'p', 'a', 'stg', 'r', 'c', 'unrealized' ]
panel_df_clean = panel_df_clean[[col for col in columns_to_keep if col in panel_df_clean.columns]]

# --- Softmax Estimation Utilities ---

def utility(w, r, scaleFactor=1000):
    w = w / scaleFactor
    if w <= 0:
        return -np.inf
    if r == 1:
        return np.log(w)
    return (w**(1 - r) - 1) / (1 - r)

def expected_utility(s, E, r, p, ks=[5, 10, 15], ch=0.15):
    pos = [utility(E + s * k, r) for k in ks]
    neg = [utility(max(E - s * k, 0), r) for k in ks]
    sum_pos = sum(pos)
    sum_neg = sum(neg)
    up = (8 * p * ch + 2 - 4 * ch) / 12
    down = (2 + 4 * ch - 8 * p * ch) / 12
    return up * sum_pos + down * sum_neg if s != 0 else utility(E, r)

def choice_prob(s_obs, E, r, p, lambda_, maxShares, scaleFactor=1000):
    utils = [lambda_ * scaleFactor * expected_utility(s, E, r, p) for s in range(maxShares + 1)]
    maxU = max(utils)
    exp_utils = [np.exp(u - maxU) for u in utils]
    denom = sum(exp_utils)
    return exp_utils[s_obs] / denom

def log_likelihood(data, r, lambda_):
    loglik = 0
    for trial in data:
        prob = choice_prob(trial['s_obs'], trial['E'], r, trial['p'], lambda_, trial['maxShares'])
        loglik += np.log(prob + 1e-12)
    return loglik

def neg_log_likelihood(params, data):
    r, lambda_ = params
    if lambda_ <= 0 or r < -5 or r > 10:
        return np.inf
    return -log_likelihood(data, r, lambda_)

def calculate_increase_probs(P_list, ch=0.15, omega=1, gamma=1, q=0.15):
    increase_probs = []
    if len(P_list) == 0:
        return increase_probs

    p = 0.5
    block_probs = [round(p, 4)]
    for i in range(1, len(P_list)):
        z = 1 if P_list[i] > P_list[i - 1] else 0
        num = ((0.5 + ch) ** z) * ((0.5 - ch) ** (1 - z)) * p
        denom = num + ((0.5 - ch) ** z) * ((0.5 + ch) ** (1 - z)) * (1 - p)
        prob = num / denom
        p = (1 - q * gamma) * prob + (q * gamma) * (1 - prob)
        block_probs.append(round(p, 4))
    return block_probs

def estimate_softmax_per_group(panel_df_clean, verbose=True, store_inputs=True):
    results = []
    grouped = panel_df_clean.groupby(['ResponseId', 'stg'])

    for (resp_id, stg), group in grouped:
        if verbose:
            print(f"⏳ Estimating for ResponseId={resp_id}, stg={stg}...")

        group = group.sort_values('r')
        s_obs_list = group['a'].tolist()
        E_list = (group['c'] + group['unrealized']).tolist()
        P_list = group['p'].tolist()

        est_r, est_lambda, ll = np.nan, np.nan, np.nan
        increase_probs = []

        if all(e == 0 for e in E_list) or all(s == 0 for s in s_obs_list) or len(group) < 2:
            if verbose:
                print("⚠️ Skipped: Degenerate E or a values.")
        else:
            try:
                increase_probs = calculate_increase_probs(P_list)
            except Exception as e:
                if verbose:
                    print(f"❌ Error in calculate_increase_probs: {e}")
                increase_probs = [0.5] * len(P_list)

            min_len = min(len(s_obs_list), len(E_list), len(P_list), len(increase_probs))
            if min_len < 2:
                if verbose:
                    print(f"⚠️ Skipped: insufficient data after aligning lists (length={min_len}).")
            else:
                s_obs_list = s_obs_list[:min_len]
                E_list = E_list[:min_len]
                P_list = P_list[:min_len]
                increase_probs = increase_probs[:min_len]

                trials = []
                for s_obs, E, price, p in zip(s_obs_list, E_list, P_list, increase_probs):
                    if price <= 0:
                        if verbose:
                            print(f"⚠️ Skipped trial with price={price}")
                        continue
                    max_shares = floor(E / price)
                    trials.append({'E': E, 's_obs': int(s_obs), 'p': p, 'maxShares': max_shares})

                if len(trials) == 0:
                    if verbose:
                        print("⚠️ Skipped: No usable trials.")
                else:
                    try:
                        result = minimize(
                            neg_log_likelihood,
                            x0=[-1.5, 2.0],
                            args=(trials,),
                            method='L-BFGS-B',
                            bounds=[(-5, 10), (0.0001, 20)]
                        )
                        if result.success:
                            est_r, est_lambda = result.x
                            ll = -result.fun
                            if verbose:
                                print(f"✅ Success: r={est_r:.3f}, lambda={est_lambda:.3f}, LL={ll:.2f}")
                        else:
                            if verbose:
                                print("❌ Optimization failed.")
                    except Exception as e:
                        if verbose:
                            print(f"❌ Estimation error: {e}")

        results.append({
            'ResponseId': resp_id,
            'stg': stg,
            'estimated_r': est_r,
            'estimated_lambda': est_lambda,
            'log_likelihood': ll,
            'E_list': E_list if store_inputs else None,
            's_obs_list': s_obs_list if store_inputs else None,
            'P_list': P_list if store_inputs else None,
            'increase_probs': increase_probs if store_inputs else None,
        })

    return pd.DataFrame(results)

# Example usage:
results_df = estimate_softmax_per_group(panel_df_clean)