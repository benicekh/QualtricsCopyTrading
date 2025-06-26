# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:02:58 2024

@author: benja
"""

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind
from statsmodels.nonparametric.smoothers_lowess import lowess

# Define the folder containing the CSV files
#folder_path = 'D:\Surfdrive\Projects\Copy Trading\Results'
#
folder_paths = [
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Results\Pilot",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Results\Pilot"
]

#Automatically select the first existing folder
folder_path = next((path for path in folder_paths if os.path.exists(path)), None)

if folder_path is None:
    raise FileNotFoundError("None of the specified folder paths exist.")

print(f"Using folder path: {folder_path}")

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


panel_df_clean = panel_df[panel_df['phaseName'] != 'training']
panel_df_clean.drop(columns=['LocationLatitude', 'LocationLongitude', 'IPAddress', 'TLs', 'treatment'], inplace=True)
panel_df_clean['copiedRank'] = panel_df_clean['TLrank'].str.extract(r'rank_(\d+)').astype(int)
panel_df_clean['copiedCRRA'] = panel_df_clean['tradeLeader'].str.extract(r'CRRA_(-?\d+\.?\d*)').astype(float)
panel_df_clean['CRRA'] = panel_df_clean['CRRA'].astype(float)
panel_df_clean['phaseReturn'] = panel_df_clean['phaseReturn'].astype(float)
panel_df_clean['nameTreatment'] = panel_df_clean['nameTreatment'].astype(int)
panel_df_clean['followersTreatment'] = panel_df_clean['followersTreatment'].astype(int)
panel_df_clean['overwrittenLottery'] = (panel_df_clean['copiedCRRA'] < panel_df_clean['CRRA']).astype(int)

crra_rank = {-1.5: 1, 0: 2, 1: 3, 3: 4, 6: 5}

# Map both columns to their ranks
panel_df_clean['CRRA_rank'] = panel_df_clean['CRRA'].map(crra_rank)
panel_df_clean['copiedCRRA_rank'] = panel_df_clean['copiedCRRA'].map(crra_rank)

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

# Outer merge on Participant id and PROLIFIC_PID
merged_df = final_dataframe.merge(
    panel_df_clean,
    left_on="Participant id",
    right_on="PROLIFIC_PID",
    how="outer",
    indicator=True  # adds a column to indicate the source of each row
)


merged_df = merged_df[merged_df["_merge"] == "both"]
columns_to_drop = ["_merge", "PROLIFIC_PID", "RecipientLastName", "RecipientFirstName", "RecipientEmail", "ExternalReference", "High", "Refresh", "Q3_3_TEXT", "DistributionChannel", "Progress", "Status_y", "Status_x", "Custom study tncs accepted at" ]  # replace with actual column names
merged_df = merged_df.drop(columns=columns_to_drop)

panel_df_clean = merged_df

# Calculate the difference
# Rank of copied - own rank
panel_df_clean['CRRAchange'] = panel_df_clean['copiedCRRA_rank'] - panel_df_clean['CRRA_rank']

average_by_combo = panel_df_clean.groupby(['nameTreatment', 'followersTreatment'])['overwrittenLottery'].mean()
print(average_by_combo)

average_by_triple = panel_df_clean.groupby(['nameTreatment', 'followersTreatment', 'phase'])['overwrittenLottery'].mean()
print(average_by_triple)

average_by_quadruple = panel_df_clean.groupby(['nameTreatment', 'followersTreatment', 'pathseries', 'phase'])['overwrittenLottery'].mean()
print(average_by_quadruple)


average_by_combo = panel_df_clean.groupby(['nameTreatment', 'followersTreatment'])['copiedRank'].mean()
print(average_by_combo)

average_by_triple = panel_df_clean.groupby(['nameTreatment', 'followersTreatment', 'phase'])['copiedRank'].mean()
print(average_by_triple)

average_by_quadruple = panel_df_clean.groupby(['nameTreatment', 'followersTreatment', 'pathseries', 'phase'])['copiedRank'].mean()
print(average_by_quadruple)


table_avg_rank = panel_df_clean.pivot_table(
    values='copiedRank',
    index='phase',
    columns=['pathseries', 'nameTreatment', 'followersTreatment'],
    aggfunc='mean'
)

table_avg_overwritten = panel_df_clean.pivot_table(
    values='overwrittenLottery',
    index='phase',
    columns=['pathseries', 'nameTreatment', 'followersTreatment'],
    aggfunc='mean'
)

table_avg_gain = panel_df_clean.pivot_table(
    values='gain',
    index='phase',
    columns=['pathseries', 'nameTreatment', 'followersTreatment'],
    aggfunc='mean'
)

table_avg_agg_gain = panel_df_clean.pivot_table(
    values='gainAll',
    index='phase',
    columns=['pathseries', 'nameTreatment', 'followersTreatment'],
    aggfunc='mean'
)

table_avg_riskrank = panel_df_clean.pivot_table(
    values='copiedCRRA_rank',
    index='phase',
    columns=['pathseries', 'nameTreatment', 'followersTreatment'],
    aggfunc='mean'
)


### cumulative rank over phases
# Create treatment labels
df = panel_df_clean.copy()
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Ensure sorting
df = df.sort_values(by=['Participant id', 'phase'])

# Compute cumulative copiedRank per participant
df['cumulativeRank'] = df.groupby('Participant id')['copiedRank'].cumsum()

# Function to get average cumulative rank per phase per treatment
def average_cumulative_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel', 'Participant id'])['cumulativeRank'].last().reset_index()
    averaged = grouped.groupby(['phase', 'TreatmentLabel'])['cumulativeRank'].mean().reset_index()
    pivot = averaged.pivot(index='phase', columns='TreatmentLabel', values='cumulativeRank')
    return pivot

# Compute data for each pathseries and combined
cumulative_dfs = {}
for path in sorted(df['pathseries'].unique()):
    cumulative_dfs[path] = average_cumulative_by_phase(df[df['pathseries'] == path])
cumulative_combined = average_cumulative_by_phase(df)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})

# First row: individual pathseries plots
for idx, path in enumerate(sorted(cumulative_dfs)):
    ax = axs[0, idx]
    for col in cumulative_dfs[path].columns:
        ax.plot(cumulative_dfs[path].index, cumulative_dfs[path][col], label=col)
    ax.set_title(f'Pathseries {path}')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Avg Cumulative copiedRank')
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined plot
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in cumulative_combined.columns:
    ax_combined.plot(cumulative_combined.index, cumulative_combined[col], label=col)
ax_combined.set_title('Combined Pathseries')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Avg Cumulative copiedRank')
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\avg_cumulative_rank_combined_plot.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"Figure saved to:\n{output_path}")

### cumulative risk rank over phases
df = panel_df_clean.copy()
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Ensure sorting
df = df.sort_values(by=['Participant id', 'phase'])

# Compute cumulative copiedRank per participant
df['cumulativeRank'] = df.groupby('Participant id')['copiedCRRA_rank'].cumsum()

# Function to get average cumulative rank per phase per treatment
def average_cumulative_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel', 'Participant id'])['cumulativeRank'].last().reset_index()
    averaged = grouped.groupby(['phase', 'TreatmentLabel'])['cumulativeRank'].mean().reset_index()
    pivot = averaged.pivot(index='phase', columns='TreatmentLabel', values='cumulativeRank')
    return pivot

# Compute data for each pathseries and combined
cumulative_dfs = {}
for path in sorted(df['pathseries'].unique()):
    cumulative_dfs[path] = average_cumulative_by_phase(df[df['pathseries'] == path])
cumulative_combined = average_cumulative_by_phase(df)

# Find global min and max across all
all_values = pd.concat([cumulative_combined] + list(cumulative_dfs.values()))
ymin = all_values.min().min()
ymax = all_values.max().max()

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})

# First row: individual pathseries plots
for idx, path in enumerate(sorted(cumulative_dfs)):
    ax = axs[0, idx]
    for col in cumulative_dfs[path].columns:
        ax.plot(cumulative_dfs[path].index, cumulative_dfs[path][col], label=col)
    ax.set_title(f'Pathseries {path}')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Avg cumulative copied risk rank')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined plot
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in cumulative_combined.columns:
    ax_combined.plot(cumulative_combined.index, cumulative_combined[col], label=col)
ax_combined.set_title('Combined Pathseries')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Avg cumulative copied risk rank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\avg_cumulative_risk_rank_combined_plot_yaligned.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✅ Figure with aligned y-axis saved to:\n{output_path}")


save_paths = [
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Results",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Results"
]

# Automatically select the first existing folder
save_path = next((path for path in save_paths if os.path.exists(path)), None)

if save_path is None:
    raise FileNotFoundError("None of the specified folder paths exist.")

# Now build full file paths
panel_data_path = os.path.join(save_path, "panel_data.csv")
panel_data_cleaned_path = os.path.join(save_path, "panel_data_cleaned.csv")

# Export the DataFrames
panel_df.to_csv(panel_data_path, index=False)
panel_df_clean.to_csv(panel_data_cleaned_path, index=False)

#### copied rank vs copied crra rank smoothed
df = panel_df_clean.copy()
df = df.sort_values(by=['Participant id', 'phase'])

# Create treatment labels
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Compute cumulative ranks per participant
df['cumulative_copiedRank'] = df.groupby('Participant id')['copiedRank'].cumsum()
df['cumulative_copiedCRRA_rank'] = df.groupby('Participant id')['copiedCRRA_rank'].cumsum()

# Function to compute average cumulative values per phase and treatment
def average_cumulative_by_phase(data, col):
    grouped = data.groupby(['phase', 'TreatmentLabel', 'Participant id'])[col].last().reset_index()
    averaged = grouped.groupby(['phase', 'TreatmentLabel'])[col].mean().reset_index()
    pivot = averaged.pivot(index='phase', columns='TreatmentLabel', values=col)
    return pivot

# Compute data
cumulative_rank_dfs = {}
cumulative_crra_dfs = {}
for path in sorted(df['pathseries'].unique()):
    cumulative_rank_dfs[path] = average_cumulative_by_phase(df[df['pathseries'] == path], 'cumulative_copiedRank')
    cumulative_crra_dfs[path] = average_cumulative_by_phase(df[df['pathseries'] == path], 'cumulative_copiedCRRA_rank')
cumulative_combined_rank = average_cumulative_by_phase(df, 'cumulative_copiedRank')
cumulative_combined_crra = average_cumulative_by_phase(df, 'cumulative_copiedCRRA_rank')

# Find global min and max for y-axis
all_values = pd.concat([cumulative_combined_rank, cumulative_combined_crra] + 
                       [v for d in zip(cumulative_rank_dfs.values(), cumulative_crra_dfs.values()) for v in d])
ymin = all_values.min().min()
ymax = all_values.max().max()

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})

# First row: per pathseries
for idx, path in enumerate(sorted(cumulative_rank_dfs)):
    ax = axs[0, idx]
    data_rank = cumulative_rank_dfs[path]
    data_crra = cumulative_crra_dfs[path]
    for col in data_rank.columns:
        smoothed_rank = lowess(data_rank[col], data_rank.index, frac=0.4, return_sorted=False)
        smoothed_crra = lowess(data_crra[col], data_crra.index, frac=0.4, return_sorted=False)
        ax.plot(data_rank.index, smoothed_rank, label=f'{col} (Rank)', linestyle='-')
        ax.plot(data_crra.index, smoothed_crra, label=f'{col} (Risk)', linestyle='--')
    ax.set_title(f'Cumulative Displayed vs Risk Rank (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Avg Cumulative Value')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
data_rank_combined = cumulative_combined_rank
data_crra_combined = cumulative_combined_crra
for col in data_rank_combined.columns:
    smoothed_rank = lowess(data_rank_combined[col], data_rank_combined.index, frac=0.4, return_sorted=False)
    smoothed_crra = lowess(data_crra_combined[col], data_crra_combined.index, frac=0.4, return_sorted=False)
    ax_combined.plot(data_rank_combined.index, smoothed_rank, label=f'{col} (Rank)', linestyle='-')
    ax_combined.plot(data_crra_combined.index, smoothed_crra, label=f'{col} (Risk)', linestyle='--')
ax_combined.set_title('Cumulative Displayed vs Risk Rank (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Avg Cumulative Value')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\smoothed_copiedRank_vs_copiedCRRA_rank.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✅ Smoothed comparison plot saved to:\n{output_path}")
#### Graphs ####

# Filter relevant data
df_filtered = panel_df_clean[
    panel_df_clean["phase"].isin([0, 4, 9]) &
    panel_df_clean["pathseries"].isin([2, 6])
].copy()

# Create treatment group labels
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}

# Create the TreatmentGroup label
df_filtered['TreatmentGroup'] = (
    df_filtered['nameTreatment'].map(name_map) + '+' +
    df_filtered['followersTreatment'].map(followers_map)
)

# Set up the 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharey=True)
fig.suptitle("Risk rank by Treatment Group for Each Phase and Pathseries", fontsize=18)

# Phases and columns
phases = [0, 4, 9]
path_cols = [2, 6, "both"]

for row_idx, phase_val in enumerate(phases):
    for col_idx, path_val in enumerate(path_cols):
        if path_val == "both":
            subset = df_filtered[df_filtered["phase"] == phase_val]
        else:
            subset = df_filtered[
                (df_filtered["phase"] == phase_val) &
                (df_filtered["pathseries"] == path_val)
            ]
        sns.boxplot(
            data=subset,
            x="TreatmentGroup",
            y="copiedCRRA_rank",
            ax=axes[row_idx][col_idx]
        )
        axes[row_idx][col_idx].set_title(
            f"phase = {phase_val}, pathseries = {path_val}"
        )
        axes[row_idx][col_idx].set_xlabel("Treatment Group (name + followers)")
        axes[row_idx][col_idx].set_ylabel("Risk Rank")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Set up the 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharey=True)
fig.suptitle("Rank by Treatment Group for Each Phase and Pathseries", fontsize=18)

# Phases and columns
phases = [0, 4, 9]
path_cols = [2, 6, "both"]

for row_idx, phase_val in enumerate(phases):
    for col_idx, path_val in enumerate(path_cols):
        if path_val == "both":
            subset = df_filtered[df_filtered["phase"] == phase_val]
        else:
            subset = df_filtered[
                (df_filtered["phase"] == phase_val) &
                (df_filtered["pathseries"] == path_val)
            ]
        sns.boxplot(
            data=subset,
            x="TreatmentGroup",
            y="copiedRank",
            ax=axes[row_idx][col_idx]
        )
        axes[row_idx][col_idx].set_title(
            f"phase = {phase_val}, pathseries = {path_val}"
        )
        axes[row_idx][col_idx].set_xlabel("Treatment Group (name + followers)")
        axes[row_idx][col_idx].set_ylabel("Rank")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Set up the 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharey=True)
fig.suptitle("Cumulative Gain by Treatment Group for Each Phase and Pathseries", fontsize=18)

# Phases and columns
phases = [0, 4, 9]
path_cols = [2, 6, "both"]

for row_idx, phase_val in enumerate(phases):
    for col_idx, path_val in enumerate(path_cols):
        if path_val == "both":
            subset = df_filtered[df_filtered["phase"] == phase_val]
        else:
            subset = df_filtered[
                (df_filtered["phase"] == phase_val) &
                (df_filtered["pathseries"] == path_val)
            ]
        sns.boxplot(
            data=subset,
            x="TreatmentGroup",
            y="gainAll",
            ax=axes[row_idx][col_idx]
        )
        axes[row_idx][col_idx].set_title(
            f"phase = {phase_val}, pathseries = {path_val}"
        )
        axes[row_idx][col_idx].set_xlabel("Treatment Group (name + followers)")
        axes[row_idx][col_idx].set_ylabel("Cumulative Gain")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Set up the 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharey=True)
fig.suptitle("Cumulative Gain by Treatment Group for Each Phase and Pathseries", fontsize=18)

# Phases and columns
phases = [0, 4, 9]
path_cols = [2, 6, "both"]

for row_idx, phase_val in enumerate(phases):
    for col_idx, path_val in enumerate(path_cols):
        if path_val == "both":
            subset = df_filtered[df_filtered["phase"] == phase_val]
        else:
            subset = df_filtered[
                (df_filtered["phase"] == phase_val) &
                (df_filtered["pathseries"] == path_val)
            ]
        sns.boxplot(
            data=subset,
            x="TreatmentGroup",
            y="gainAll",
            ax=axes[row_idx][col_idx]
        )
        axes[row_idx][col_idx].set_title(
            f"phase = {phase_val}, pathseries = {path_val}"
        )
        axes[row_idx][col_idx].set_xlabel("Treatment Group (name + followers)")
        axes[row_idx][col_idx].set_ylabel("Cumulative Gain")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Set up the 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharey=True)
fig.suptitle("Risk Falk by Treatment Group for Each Phase and Pathseries", fontsize=18)

df_filtered['RiskFalk'] = panel_df_clean['RiskFalk'].astype(int)
# Phases and columns
phases = [0, 4, 9]
path_cols = [2, 6, "both"]

for row_idx, phase_val in enumerate(phases):
    for col_idx, path_val in enumerate(path_cols):
        if path_val == "both":
            subset = df_filtered[df_filtered["phase"] == phase_val]
        else:
            subset = df_filtered[
                (df_filtered["phase"] == phase_val) &
                (df_filtered["pathseries"] == path_val)
            ]
        sns.boxplot(
            data=subset,
            x="TreatmentGroup",
            y="RiskFalk",
            ax=axes[row_idx][col_idx]
        )
        axes[row_idx][col_idx].set_title(
            f"phase = {phase_val}, pathseries = {path_val}"
        )
        axes[row_idx][col_idx].set_xlabel("Treatment Group (name + followers)")
        axes[row_idx][col_idx].set_ylabel("Falk risk 1-11")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Set up the 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharey=True)
fig.suptitle("Lottery Choice by Treatment Group for Each Phase and Pathseries", fontsize=18)

# Phases and columns
phases = [0, 4, 9]
path_cols = [2, 6, "both"]

for row_idx, phase_val in enumerate(phases):
    for col_idx, path_val in enumerate(path_cols):
        if path_val == "both":
            subset = df_filtered[df_filtered["phase"] == phase_val]
        else:
            subset = df_filtered[
                (df_filtered["phase"] == phase_val) &
                (df_filtered["pathseries"] == path_val)
            ]
        sns.boxplot(
            data=subset,
            x="TreatmentGroup",
            y="CRRA_rank",
            ax=axes[row_idx][col_idx]
        )
        axes[row_idx][col_idx].set_title(
            f"phase = {phase_val}, pathseries = {path_val}"
        )
        axes[row_idx][col_idx].set_xlabel("Treatment Group (name + followers)")
        axes[row_idx][col_idx].set_ylabel("CRRA_rank 1=highest risk, 5=lowest risk")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Set up the 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharey=True)
fig.suptitle("Risk rank deviation by Treatment Group for Each Phase and Pathseries", fontsize=18)

# Phases and columns
phases = [0, 4, 9]
path_cols = [2, 6, "both"]

for row_idx, phase_val in enumerate(phases):
    for col_idx, path_val in enumerate(path_cols):
        if path_val == "both":
            subset = df_filtered[df_filtered["phase"] == phase_val]
        else:
            subset = df_filtered[
                (df_filtered["phase"] == phase_val) &
                (df_filtered["pathseries"] == path_val)
            ]
        sns.boxplot(
            data=subset,
            x="TreatmentGroup",
            y="CRRAchange",
            ax=axes[row_idx][col_idx]
        )
        axes[row_idx][col_idx].set_title(
            f"phase = {phase_val}, pathseries = {path_val}"
        )
        axes[row_idx][col_idx].set_xlabel("Treatment Group (name + followers)")
        axes[row_idx][col_idx].set_ylabel("CRRA rank deviation")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


### Average number of change of copied bot
df = panel_df_clean.copy()
df = df.sort_values(by=['Participant id', 'phase'])

# Create treatment labels
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Calculate change flag per row (whether participant switched compared to previous phase)
df['rank_change'] = df.groupby('Participant id')['copiedCRRA_rank'].transform(lambda x: x != x.shift(1)).astype(int)

# Calculate cumulative change count per participant over phases
df['cumulative_rank_change'] = df.groupby('Participant id')['rank_change'].cumsum()

# Function to compute average cumulative changes per phase and treatment
def average_cumulative_changes_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel', 'Participant id'])['cumulative_rank_change'].last().reset_index()
    averaged = grouped.groupby(['phase', 'TreatmentLabel'])['cumulative_rank_change'].mean().reset_index()
    pivot = averaged.pivot(index='phase', columns='TreatmentLabel', values='cumulative_rank_change')
    return pivot

# Compute data for each pathseries and combined
average_cumulative_dfs = {}
for path in sorted(df['pathseries'].unique()):
    average_cumulative_dfs[path] = average_cumulative_changes_by_phase(df[df['pathseries'] == path])
average_cumulative_combined = average_cumulative_changes_by_phase(df)

# Find global min and max across all
all_values = pd.concat([average_cumulative_combined] + list(average_cumulative_dfs.values()))
ymin = all_values.min().min()
ymax = all_values.max().max()

# Plotting setup
fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})

# First row: per pathseries
for idx, path in enumerate(sorted(average_cumulative_dfs)):
    ax = axs[0, idx]
    for col in average_cumulative_dfs[path].columns:
        ax.plot(average_cumulative_dfs[path].index, average_cumulative_dfs[path][col], label=col)
    ax.set_title(f'Cumulative Copy Changes (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Avg Cumulative Changes')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined plot
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_cumulative_combined.columns:
    ax_combined.plot(average_cumulative_combined.index, average_cumulative_combined[col], label=col)
ax_combined.set_title('Cumulative Copy Changes (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Avg Cumulative Changes')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\cumulative_avg_changes_over_time_yaligned.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✅ Figure with aligned y-axis saved to:\n{output_path}")

### Average cumulative rank
df = panel_df_clean.copy()
df = df.sort_values(by=['Participant id', 'phase'])

# Create treatment labels
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Compute cumulative copiedRank per participant
df['cumulativeRank'] = df.groupby('Participant id')['copiedRank'].cumsum()

# Function to compute average cumulative copiedRank per phase and treatment
def average_cumulative_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel', 'Participant id'])['cumulativeRank'].last().reset_index()
    averaged = grouped.groupby(['phase', 'TreatmentLabel'])['cumulativeRank'].mean().reset_index()
    pivot = averaged.pivot(index='phase', columns='TreatmentLabel', values='cumulativeRank')
    return pivot

# Compute data for each pathseries and combined
average_cumulative_dfs = {}
for path in sorted(df['pathseries'].unique()):
    average_cumulative_dfs[path] = average_cumulative_rank_by_phase(df[df['pathseries'] == path])
average_cumulative_combined = average_cumulative_rank_by_phase(df)

# Find global min and max across all
all_values = pd.concat([average_cumulative_combined] + list(average_cumulative_dfs.values()))
ymin = all_values.min().min()
ymax = all_values.max().max()

# Plotting setup
fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})

# First row: per pathseries
for idx, path in enumerate(sorted(average_cumulative_dfs)):
    ax = axs[0, idx]
    for col in average_cumulative_dfs[path].columns:
        ax.plot(average_cumulative_dfs[path].index, average_cumulative_dfs[path][col], label=col)
    ax.set_title(f'Cumulative Displayed Rank (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Avg Cumulative copiedRank')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined plot
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_cumulative_combined.columns:
    ax_combined.plot(average_cumulative_combined.index, average_cumulative_combined[col], label=col)
ax_combined.set_title('Cumulative Displayed Rank (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Avg Cumulative copiedRank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\cumulative_avg_copiedrank_over_time_yaligned.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✅ Figure with aligned y-axis (cumulative copiedRank) saved to:\n{output_path}")


### Average streaks and switiching

df = panel_df_clean.copy()
df = df.sort_values(by=['Participant id', 'phase'])
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Helper: calculate streak lengths per participant
def compute_streak_lengths(x):
    return x.groupby((x != x.shift()).cumsum()).transform('count')

# Add streak lengths and switch size per participant
df['streak_length'] = df.groupby('Participant id')['copiedCRRA_rank'].transform(compute_streak_lengths)
df['crra_diff'] = df.groupby('Participant id')['copiedCRRA_rank'].diff()
df['rank_diff'] = df.groupby('Participant id')['copiedRank'].diff()

# Function: calculate average streak and switches per treatment
def summarize_all(data):
    # Streaks
    streaks = data.groupby(['Participant id'])['streak_length'].mean().reset_index(name='avg_streak')
    # Switches CRRA
    switches_crra = data.loc[data['crra_diff'].notna() & (data['crra_diff'] != 0)]
    switches_crra_summary = switches_crra.groupby('Participant id')['crra_diff'].mean().reset_index(name='avg_switch_crra')
    # Switches displayed rank
    switches_rank = data.loc[data['rank_diff'].notna() & (data['rank_diff'] != 0)]
    switches_rank_summary = switches_rank.groupby('Participant id')['rank_diff'].mean().reset_index(name='avg_switch_rank')

    # Merge treatment info
    treatments = data.drop_duplicates('Participant id')[['Participant id', 'TreatmentLabel']]
    streaks = streaks.merge(treatments, on='Participant id')
    switches_crra_summary = switches_crra_summary.merge(treatments, on='Participant id')
    switches_rank_summary = switches_rank_summary.merge(treatments, on='Participant id')

    # Group by treatment
    avg_streaks = streaks.groupby('TreatmentLabel')['avg_streak'].mean().round(2)
    avg_switches_crra = switches_crra_summary.groupby('TreatmentLabel')['avg_switch_crra'].mean().round(2)
    avg_switches_rank = switches_rank_summary.groupby('TreatmentLabel')['avg_switch_rank'].mean().round(2)

    # Combine into single table
    combined = pd.concat([avg_streaks, avg_switches_crra, avg_switches_rank], axis=1)
    combined.columns = [
        'Avg Streak Length',
        'Avg Switch CRRA (↑ = less risk)',
        'Avg Switch Rank (↑ = higher rank)'
    ]

    return combined

# Collect results
results = {}
results['Combined'] = summarize_all(df)

for path in sorted(df['pathseries'].unique()):
    results[f'Pathseries {path}'] = summarize_all(df[df['pathseries'] == path])

# Plotting all as tables in one figure
fig, axs = plt.subplots(len(results), 1, figsize=(12, 3.5 * len(results)))
if len(results) == 1:
    axs = [axs]

for ax, (title, table_df) in zip(axs, results.items()):
    ax.axis('off')
    table = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title(title, fontweight='bold', pad=10)

plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\streaks_and_switches_summary_with_rank_corrected.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✅ Combined table with corrected rank label saved to:\n{output_path}")


#### Falk Scatterplot
# Prepare and clean data
df = panel_df_clean.copy()
df['RiskFalk'] = pd.to_numeric(df['RiskFalk'], errors='coerce')
df['copiedCRRA_rank'] = pd.to_numeric(df['copiedCRRA_rank'], errors='coerce')

# Calculate average copiedCRRA_rank per participant
avg_copied = df.groupby(['Participant id', 'pathseries']).agg({
    'copiedCRRA_rank': 'mean',
    'RiskFalk': 'first'  # assuming RiskFalk doesn't change per participant
}).reset_index()

# Plot
plt.figure(figsize=(10, 6))
for path in sorted(avg_copied['pathseries'].unique()):
    subset = avg_copied[avg_copied['pathseries'] == path]
    plt.scatter(subset['RiskFalk'], subset['copiedCRRA_rank'], label=f'Pathseries {path}', alpha=0.7)

plt.xlabel('RiskFalk')
plt.ylabel('Average copiedCRRA_rank')
plt.title('Average copiedCRRA_rank vs. RiskFalk by Pathseries')
plt.legend(title='Pathseries')
plt.grid(True)
plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\scatter_riskfalk_vs_copiedcrrarank.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✅ Scatterplot saved to:\n{output_path}")


#### Lottery choice scatter
df = panel_df_clean.copy()
df['copiedCRRA_rank'] = pd.to_numeric(df['copiedCRRA_rank'], errors='coerce')
df['CRRA_rank'] = pd.to_numeric(df['CRRA_rank'], errors='coerce')  # assumed column name; adjust if needed

# Calculate average copiedCRRA_rank per participant
avg_copied = df.groupby(['Participant id', 'pathseries']).agg({
    'copiedCRRA_rank': 'mean',
    'CRRA_rank': 'first'  # assuming CRRA_rank doesn't change per participant
}).reset_index()

# Plot
plt.figure(figsize=(10, 6))
for path in sorted(avg_copied['pathseries'].unique()):
    subset = avg_copied[avg_copied['pathseries'] == path]
    plt.scatter(subset['CRRA_rank'], subset['copiedCRRA_rank'], label=f'Pathseries {path}', alpha=0.7)

plt.xlabel('CRRA_rank (own risk rank)')
plt.ylabel('Average copiedCRRA_rank')
plt.title('Average copiedCRRA_rank vs. CRRA_rank by Pathseries')
plt.legend(title='Pathseries')
plt.grid(True)
plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\scatter_crrarank_vs_copiedcrrarank.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✅ Scatterplot saved to:\n{output_path}")

#### SCatter for phase 0
df = panel_df_clean.copy()
df['RiskFalk'] = pd.to_numeric(df['RiskFalk'], errors='coerce')
df['copiedCRRA_rank'] = pd.to_numeric(df['copiedCRRA_rank'], errors='coerce')
df['CRRA_rank'] = pd.to_numeric(df['CRRA_rank'], errors='coerce')  # assumed name

# Filter to phase 0
df_phase0 = df[df['phase'] == 0]

# Get unique participant values at phase 0
phase0_data = df_phase0.groupby(['Participant id', 'pathseries']).agg({
    'copiedCRRA_rank': 'first',
    'RiskFalk': 'first',
    'CRRA_rank': 'first'
}).reset_index()

### 1️⃣ Scatterplot: copiedCRRA_rank vs. RiskFalk
plt.figure(figsize=(10, 6))
for path in sorted(phase0_data['pathseries'].unique()):
    subset = phase0_data[phase0_data['pathseries'] == path]
    plt.scatter(subset['RiskFalk'], subset['copiedCRRA_rank'], label=f'Pathseries {path}', alpha=0.7)

plt.xlabel('RiskFalk')
plt.ylabel('copiedCRRA_rank (phase 0)')
plt.title('Phase 0 copiedCRRA_rank vs. RiskFalk by Pathseries')
plt.legend(title='Pathseries')
plt.grid(True)
plt.tight_layout()

# Save and show
output_path1 = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\scatter_riskfalk_vs_copiedcrrarank_phase0.png"
plt.savefig(output_path1, bbox_inches='tight')
plt.show()
print(f"✅ Phase 0 RiskFalk scatterplot saved to:\n{output_path1}")

### 2️⃣ Scatterplot: copiedCRRA_rank vs. CRRA_rank
plt.figure(figsize=(10, 6))
for path in sorted(phase0_data['pathseries'].unique()):
    subset = phase0_data[phase0_data['pathseries'] == path]
    plt.scatter(subset['CRRA_rank'], subset['copiedCRRA_rank'], label=f'Pathseries {path}', alpha=0.7)

plt.xlabel('CRRA_rank (own risk rank)')
plt.ylabel('copiedCRRA_rank (phase 0)')
plt.title('Phase 0 copiedCRRA_rank vs. CRRA_rank by Pathseries')
plt.legend(title='Pathseries')
plt.grid(True)
plt.tight_layout()

# Save and show
output_path2 = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\scatter_crrarank_vs_copiedcrrarank_phase0.png"
plt.savefig(output_path2, bbox_inches='tight')
plt.show()
print(f"✅ Phase 0 CRRA_rank scatterplot saved to:\n{output_path2}")

#### Risk trajectory paths
df = panel_df_clean.copy()
df['copiedCRRA_rank'] = pd.to_numeric(df['copiedCRRA_rank'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedCRRA_rank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['copiedCRRA_rank'].mean().reset_index()
    pivot = grouped.pivot(index='phase', columns='TreatmentLabel', values='copiedCRRA_rank')
    return pivot

# Compute data
average_dfs = {}
for path in sorted(df['pathseries'].unique()):
    average_dfs[path] = average_rank_by_phase(df[df['pathseries'] == path])
average_combined = average_rank_by_phase(df)

# Find global min and max for y-axis
all_values = pd.concat([average_combined] + list(average_dfs.values()))
ymin = all_values.min().min()
ymax = all_values.max().max()

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})

# First row: by pathseries
for idx, path in enumerate(sorted(average_dfs)):
    ax = axs[0, idx]
    for col in average_dfs[path].columns:
        ax.plot(average_dfs[path].index, average_dfs[path][col], label=col, marker='o')
    ax.set_title(f'Risk Trajectory (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average copiedCRRA_rank')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_combined.columns:
    ax_combined.plot(average_combined.index, average_combined[col], label=col, marker='o')
ax_combined.set_title('Risk Trajectory (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Average copiedCRRA_rank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

plt.tight_layout()

# Save and show
output_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\risk_trajectory_plots_same_yaxis.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✅ Risk trajectory plots with aligned y-axis saved to:\n{output_path}")

# Summary Stats
# Filter for phase 0
df_filtered = panel_df_clean[panel_df_clean['phase'] == 0].copy()

# Ensure numeric columns are treated correctly
df_filtered["RiskFalk"] = pd.to_numeric(df_filtered["RiskFalk"], errors="coerce")
df_filtered["copiedCRRA_rank"] = pd.to_numeric(df_filtered["copiedCRRA_rank"], errors="coerce")

# Create intuitive TreatmentGroup labels
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df_filtered["TreatmentGroup"] = (
    df_filtered["nameTreatment"].map(name_map) + '+' +
    df_filtered["followersTreatment"].map(followers_map)
)
df_filtered["comprCount"] = df_filtered["comprCount"].astype(int)

# Compute summary statistics
summary_stats = df_filtered.groupby("TreatmentGroup").agg({
    "RiskFalk": "mean",
    "copiedCRRA_rank": "mean",
    "SexCat": lambda x: (x == 1).mean(),
    "Country of residenceCat": lambda x: (x == 1).mean(),
    "Age": "mean",
    "Time taken": "median",
    "comprCount": lambda x: (x != 0).mean()  # proportion of nonzero values
}).rename(columns={
    "RiskFalk": "Avg Risk (Falk)",
    "copiedCRRA_rank": "Lottery",
    "SexCat": "% Female",
    "Country of residenceCat": "% US-based",
    "Age": "Avg Age",
    "Time taken": "Median Time Taken",
    "comprCount": "Comprehension"
})

# Round for display
summary_stats = summary_stats.round(2)

# Export to Excel for PowerPoint
summary_stats.to_excel("summary_statistics_by_treatment.xlsx")

# Optional: show it in the notebook or terminal
print(summary_stats)

fig, ax = plt.subplots(figsize=(10, 2))  # Adjust size as needed
ax.axis('off')
table = ax.table(cellText=summary_stats.values,
                 colLabels=summary_stats.columns,
                 rowLabels=summary_stats.index,
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.savefig("summary_stats_table.png", bbox_inches='tight')


## AVG rank table
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}

# Copy the original DataFrame to avoid modifying it directly
df = panel_df_clean.copy()

# Create treatment label column
df["TreatmentLabel"] = df["nameTreatment"].map(name_map) + '+' + df["followersTreatment"].map(followers_map)

# Step 2: Create pivot table with new labels
table_avg_rank = df.pivot_table(
    values='copiedRank',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
).round(2)

# Step 3: Plot and save as PNG
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed
ax.axis('off')
table = ax.table(
    cellText=table_avg_rank.values,
    rowLabels=table_avg_rank.index,
    colLabels=[f"{p}-{t}" for p, t in table_avg_rank.columns],
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Save to PNG
plt.savefig("average_rank_table.png", bbox_inches='tight')
plt.close()

print("Table saved as 'average_rank_table.png'")

## AVG copy risk rank table
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}

# Copy the original DataFrame to avoid modifying it directly
df = panel_df_clean.copy()

# Create treatment label column
df["TreatmentLabel"] = df["nameTreatment"].map(name_map) + '+' + df["followersTreatment"].map(followers_map)

# Step 2: Create pivot table with new labels
table_avg_riskrank = df.pivot_table(
    values='copiedCRRA_rank',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
).round(2)

# Step 3: Plot and save as PNG
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed
ax.axis('off')
table = ax.table(
    cellText=table_avg_rank.values,
    rowLabels=table_avg_rank.index,
    colLabels=[f"{p}-{t}" for p, t in table_avg_rank.columns],
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Save to PNG
plt.savefig("average_risk_rank_table.png", bbox_inches='tight')
plt.close()

## AVG copy risk rank deviation table
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}

# Copy the original DataFrame to avoid modifying it directly
df = panel_df_clean.copy()

# Create treatment label column
df["TreatmentLabel"] = df["nameTreatment"].map(name_map) + '+' + df["followersTreatment"].map(followers_map)

# Step 2: Create pivot table with new labels
table_avg_riskrank_change = df.pivot_table(
    values='CRRAchange',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
).round(2)

# Step 3: Plot and save as PNG
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed
ax.axis('off')
table = ax.table(
    cellText=table_avg_rank.values,
    rowLabels=table_avg_rank.index,
    colLabels=[f"{p}-{t}" for p, t in table_avg_rank.columns],
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Save to PNG
plt.savefig("average_risk_rank_table.png", bbox_inches='tight')
plt.close()