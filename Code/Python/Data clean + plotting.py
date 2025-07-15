import os
import pandas as pd
import seaborn as sns
import re
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind
from statsmodels.nonparametric.smoothers_lowess import lowess

### SET PATHS ###
results_paths = [
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Results\Pilot",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Results\Pilot"
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
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Botdata",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Botdata"
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
panel_df_clean['CRRA_rank'] = panel_df_clean['CRRA'].map(crra_rank)
panel_df_clean['copiedCRRA_rank'] = panel_df_clean['copiedCRRA'].map(crra_rank)

dataframes_list = []


for idx, filename in enumerate(os.listdir(demo_path)):
    if filename.endswith('.csv'):  
        demo_file_path = os.path.join(demo_path, filename)
        df = pd.read_csv(demo_file_path)
        
        
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

panel_df_clean['CRRAchange'] = panel_df_clean['copiedCRRA_rank'] - panel_df_clean['CRRA_rank']


# Dictionary to hold all series
all_series_data = {}

for filename in os.listdir(bot_path):
    if filename.endswith('.txt'):
        bot_file_path = os.path.join(bot_path, filename)
        
        with open(bot_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract the series number from filename
        match = re.search(r'series.*?(\d+)', filename)
        if match:
            series_number = match.group(1)
            key = f"series {series_number}"

            # Clean content
            content_cleaned = (
                content
                .replace('const data =', '')  # remove assignment
                .replace('NaN', 'float("nan")')
                .replace('true', 'True')
                .replace('false', 'False')
                .strip()
            )

            # Remove trailing '};' if exists
            if content_cleaned.endswith('};'):
                content_cleaned = content_cleaned[:-2]

            # Fix missing quotes around dictionary keys
            content_cleaned = re.sub(r'(\w+)\s*:', r'"\1":', content_cleaned)

            # Make sure it ends properly
            if not content_cleaned.endswith('}'):
                content_cleaned += '}'

            try:
                series_data = eval(content_cleaned)
                all_series_data[key] = series_data
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")

all_rows = []

# Loop over series
for series_key, series_data in all_series_data.items():
    parts = series_key.split(' ')
    if len(parts) == 2 and parts[1].isdigit():
        pathseries_number = int(parts[1])
    else:
        print(f"Skipping invalid series key: {series_key}")
        continue  # Skip if series_key is not like 'series x'
    
    # Loop over stages
    for stage_key in series_data.keys():
        if stage_key.startswith('stage_'):
            phase_number = int(stage_key.split('_')[1])
            round_data = series_data[stage_key].get('round_0', {})
            
            # Loop over ranks
            for rank_idx in range(1, 6):
                rank_key = f'rank_{rank_idx}'
                rank_data = round_data.get(rank_key, {})
                
                if rank_data:
                    row = {
                        'phase': phase_number,
                        'returnAll_bot': rank_data.get('returnAllv2'),
                        'return_bot': rank_data.get('phaseReturn'),
                        'wealth_bot': rank_data.get('wealthAllv2'),
                        'tradeLeader': rank_data.get('ResponseId'),
                        'path': rank_data.get('path'),
                        'returnAllv2': rank_data.get('returnAllv2'),
                        'pathseries': pathseries_number,
                        'rank': rank_idx  # extract from rank_x
                    }
                    all_rows.append(row)
                    
            # Add Final rows from stage_9 → round_40
            if stage_key == 'stage_9':
                final_round = series_data[stage_key].get('round_40', {})
                for rank_idx in range(1, 6):
                    rank_key = f'rank_{rank_idx}'
                    rank_data = final_round.get(rank_key, {})
                    
                    if rank_data:
                        row = {
                            'phase': 'Final',
                           'returnAll_bot': rank_data.get('returnAllv2'),
                           'return_bot': rank_data.get('phaseReturn'),
                           'wealth_bot': rank_data.get('wealthAllv2'),
                            'tradeLeader': rank_data.get('ResponseId'),
                            'path': rank_data.get('path'),
                            'returnAllv2': rank_data.get('returnAllv2'),
                            'pathseries': pathseries_number,
                            'rank': rank_idx  # extract from rank_x
                        }
                        all_rows.append(row)

botdata_df = pd.DataFrame(all_rows)

# Calculate overall rank within phase + pathseries
botdata_df['overallRank'] = botdata_df.groupby(['pathseries', 'phase'])['returnAllv2'] \
                                       .rank(method='dense', ascending=False) \
                                       .astype(int)
botdata_df = botdata_df.sort_values(by=['pathseries', 'phase', 'overallRank']).reset_index(drop=True)
# Remove 'Final' phase rows
botdata_filtered = botdata_df[botdata_df['phase'] != 'Final'].copy()

# Make sure types are aligned
panel_df_clean['tradeLeader'] = panel_df_clean['tradeLeader'].astype(str).str.strip()
botdata_filtered['tradeLeader'] = botdata_filtered['tradeLeader'].astype(str).str.strip()

panel_df_clean['path'] = panel_df_clean['path'].astype(str).str.strip()
botdata_filtered['path'] = botdata_filtered['path'].astype(str).str.strip()

panel_df_clean['phase'] = panel_df_clean['phase'].astype(int)
botdata_filtered['phase'] = botdata_filtered['phase'].astype(int)

panel_df_clean['pathseries'] = panel_df_clean['pathseries'].astype(int)
botdata_filtered['pathseries'] = botdata_filtered['pathseries'].astype(int)

#CMCLOGIT dataframe creation
alternatives = ['CRRA_-1.5', 'CRRA_0', 'CRRA_1', 'CRRA_3', 'CRRA_6']
df = panel_df_clean.copy()
df['obs_id'] = df['Participant id'].astype(str) + "_" + df['phase'].astype(str)
cmclogit = df.loc[df.index.repeat(len(alternatives))].copy()
cmclogit['alt'] = alternatives * len(df)
cmclogit['chosen'] = (cmclogit['alt'] == cmclogit['tradeLeader']).astype(int)
cmclogit['tradeLeader'] = cmclogit['alt']
cmclogit.drop(columns=['alt'], inplace=True)
cmclogit['copiedCRRA'] = cmclogit['tradeLeader'].str.extract(r'CRRA_(-?\d+\.?\d*)').astype(float)
cmclogit['copiedCRRA_rank'] = cmclogit['copiedCRRA'].map(crra_rank)


# Merge
merged_df = panel_df_clean.merge(
    botdata_filtered,
    how='left',
    on=['tradeLeader', 'phase', 'pathseries']
)

# Merge
cmc_merged_df = cmclogit.merge(
    botdata_filtered,
    how='left',
    on=['tradeLeader', 'phase', 'pathseries']
)


merged_df = merged_df.sort_values(by=['Submission id', 'phase', 'rank']).reset_index(drop=True)
cmc_merged_df = cmc_merged_df.sort_values(by=['Submission id', 'phase', 'rank']).reset_index(drop=True)
panel_df_clean_bot = merged_df.sort_values(by=['Submission id', 'phase', 'rank']).reset_index(drop=True)
###############################################################################
df = panel_df_clean_bot[panel_df_clean_bot['phase'] == 0].copy()

# Step 1: Check distribution of CRRA_rank (1 = highest risk, 5 = lowest risk)
riskrank_counts = df['CRRA_rank'].value_counts(normalize=True).sort_index()
df["RiskFalk"] = pd.to_numeric(df["RiskFalk"], errors="coerce")
print("\nCRRA_rank distribution (proportion per category):")
print(riskrank_counts)

# Step 2: Calculate cumulative quantiles
quantiles = riskrank_counts.cumsum().values
print("\nCumulative quantiles for binning:", quantiles)

# Step 3: Handle RiskFalk ties by ranking them
df['RiskFalk_ranked'] = df['RiskFalk'].rank(method='first')

# Step 4: Bin RiskFalk into 5 categories matching CRRA_rank distribution
# We use labels [5,4,3,2,1] so the highest RiskFalk gets category 1, lowest gets 5
df['RiskFalk_cat'] = pd.qcut(
    df['RiskFalk_ranked'],
    q=[0] + list(quantiles),
    labels=[5, 4, 3, 2, 1]  # reverse the labels here
)

# Step 5: Convert to integer
df['RiskFalk_cat'] = df['RiskFalk_cat'].astype(int)

# Step 6: Drop helper column
df = df.drop(columns=['RiskFalk_ranked'])

# Step 7: Check final distribution
print("\nMapped RiskFalk categories distribution (after reversing):")
print(df['RiskFalk_cat'].value_counts().sort_index())

print("\nFinal table comparing counts:")
comparison = pd.DataFrame({
    'CRRA_rank': df['CRRA_rank'].value_counts().sort_index(),
    'RiskFalk_cat': df['RiskFalk_cat'].value_counts().sort_index()
})
print(comparison)

# === Diagnostic bar plot comparing distributions ===
plt.figure(figsize=(8, 6))
df['CRRA_rank'].value_counts(normalize=True).sort_index().plot(kind='bar', alpha=0.6, label='CRRA_rank')
df['RiskFalk_cat'].value_counts(normalize=True).sort_index().plot(kind='bar', alpha=0.6, label='RiskFalk_cat', color='orange')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.title('Comparison of CRRA_rank and RiskFalk_cat distributions')
plt.legend()
plt.tight_layout()
plt.show()
# Calculate Pearson correlation
corr_pearson = df[['CRRA_rank', 'RiskFalk_cat']].corr(method='pearson').loc['CRRA_rank', 'RiskFalk_cat']
# Calculate Spearman correlation (rank-based, nonparametric)
corr_spearman = df[['CRRA_rank', 'RiskFalk_cat']].corr(method='spearman').loc['CRRA_rank', 'RiskFalk_cat']
# Print the results
print("\nCorrelation between CRRA_rank and RiskFalk_cat:")
print(f"Pearson correlation:  {corr_pearson:.3f}")
print(f"Spearman correlation: {corr_spearman:.3f}")

#merge back
panel_df_clean_bot = panel_df_clean_bot.merge(
    df[['Submission id', 'RiskFalk_cat']],
    on='Submission id',
    how='left'
)

panel_df_clean_bot['Falkchange'] = panel_df_clean_bot['copiedCRRA_rank'] - panel_df_clean_bot['RiskFalk_cat']

panel_df_clean_bot = panel_df_clean_bot.sort_values(by=['Submission id', 'phase'])
panel_df_clean_bot['copiedCRRA_change'] = panel_df_clean_bot.groupby('Submission id')['copiedCRRA'].diff().ne(0).astype(int)
change_counts = panel_df_clean_bot.groupby('Submission id')['copiedCRRA_change'].sum() -1
panel_df_clean_bot['botChangeSum'] = panel_df_clean_bot['Submission id'].map(change_counts)
panel_df_clean_bot['botChangeSumToPhase'] = panel_df_clean_bot.groupby('Submission id')['copiedCRRA_change'].cumsum() -1

# Export the DataFrames
panel_data_path = os.path.join(cleaned_path, "panel_data.csv")
panel_data_cleaned_path = os.path.join(cleaned_path, "panel_data_cleaned.csv")
panel_data_cleaned_bot_path = os.path.join(cleaned_path, "panel_data_cleaned_bot.csv")
cmclogit_path = os.path.join(cleaned_path, "cmclogit_panel.csv")
panel_df.to_csv(panel_data_path, index=False)
panel_df_clean.to_csv(panel_data_cleaned_path, index=False)
panel_df_clean_bot.to_csv(panel_data_cleaned_bot_path, index=False)
cmc_merged_df.to_csv(cmclogit_path, index=False)

panel_df.to_excel(os.path.join(cleaned_path, "panel_data.xlsx"), index=False)
panel_df_clean.to_excel(os.path.join(cleaned_path, "panel_data_cleaned.xlsx"), index=False)
panel_df_clean_bot.to_excel(os.path.join(cleaned_path, "panel_data_cleaned_bot.xlsx"), index=False)




###############################################################################
###### Summary Stats
###############################################################################

data = panel_df_clean_bot.copy()
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
data['TreatmentLabel'] = data['nameTreatment'].map(name_map) + '+' + data['followersTreatment'].map(followers_map)

# Pivot tables with TreatmentLabel in columns
table_avg_rank = data.pivot_table(
    values='copiedRank',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
).round(2)

table_median_rank = data.pivot_table(
    values='copiedRank',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='median'
)

table_avg_overwritten = data.pivot_table(
    values='overwrittenLottery',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
)

table_median_overwritten = data.pivot_table(
    values='overwrittenLottery',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='median'
)

table_avg_gain = data.pivot_table(
    values='gain',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
)

table_avg_agg_gain = data.pivot_table(
    values='gainAll',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
)

table_avg_riskrank = data.pivot_table(
    values='copiedCRRA_rank',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
)

table_median_riskrank = data.pivot_table(
    values='copiedCRRA_rank',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='median'
)

table_avg_overallrank = data.pivot_table(
    values='overallRank',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
).round(2)

table_median_overallrank = data.pivot_table(
    values='overallRank',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='median'
).round(2)

table_avg_riskdeviation = data.pivot_table(
    values='CRRAchange',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
)

table_median_riskdeviation = data.pivot_table(
    values='CRRAchange',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='median'
)

table_avg_riskdeviation_falk = data.pivot_table(
    values='Falkchange',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
)

table_median_riskdeviation_falk = data.pivot_table(
    values='Falkchange',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='median'
)

table_avg_botchange = data.pivot_table(
    values='botChangeSumToPhase',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='mean'
)

table_median_botchange = data.pivot_table(
    values='botChangeSumToPhase',
    index='phase',
    columns=['pathseries', 'TreatmentLabel'],
    aggfunc='median'
)

# Summary Stats
# Filter for phase 0
df_filtered = panel_df_clean[panel_df_clean['phase'] == 0].copy()
df_filtered["RiskFalk"] = pd.to_numeric(df_filtered["RiskFalk"], errors="coerce")
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

###############################################################################
###### Streaks and switching
df = panel_df_clean_bot.copy()
df = df.sort_values(by=['Participant id', 'phase'])

name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Helper: calculate streak lengths per participant
def compute_streak_lengths(x):
    return x.groupby((x != x.shift()).cumsum()).transform('count')

# Add streak lengths and switch sizes per participant
df['streak_length'] = df.groupby('Participant id')['copiedCRRA_rank'].transform(compute_streak_lengths)
df['crra_diff'] = df.groupby('Participant id')['copiedCRRA_rank'].diff()
df['rank_diff'] = df.groupby('Participant id')['copiedRank'].diff()
df['overallRank_diff'] = df.groupby('Participant id')['overallRank'].diff()  # ← added this line

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
    # Switches overall rank (NEW)
    switches_overallrank = data.loc[data['overallRank_diff'].notna() & (data['overallRank_diff'] != 0)]
    switches_overallrank_summary = switches_overallrank.groupby('Participant id')['overallRank_diff'].mean().reset_index(name='avg_switch_overallrank')

    # Merge treatment info
    treatments = data.drop_duplicates('Participant id')[['Participant id', 'TreatmentLabel']]
    streaks = streaks.merge(treatments, on='Participant id')
    switches_crra_summary = switches_crra_summary.merge(treatments, on='Participant id')
    switches_rank_summary = switches_rank_summary.merge(treatments, on='Participant id')
    switches_overallrank_summary = switches_overallrank_summary.merge(treatments, on='Participant id')

    # Group by treatment
    avg_streaks = streaks.groupby('TreatmentLabel')['avg_streak'].mean().round(2)
    avg_switches_crra = switches_crra_summary.groupby('TreatmentLabel')['avg_switch_crra'].mean().round(2)
    avg_switches_rank = switches_rank_summary.groupby('TreatmentLabel')['avg_switch_rank'].mean().round(2)
    avg_switches_overallrank = switches_overallrank_summary.groupby('TreatmentLabel')['avg_switch_overallrank'].mean().round(2)

    # Combine into single table
    combined = pd.concat([avg_streaks, avg_switches_crra, avg_switches_rank, avg_switches_overallrank], axis=1)
    combined.columns = [
        'Avg Streak Length',
        'Avg Switch CRRA (↑ = less risk)',
        'Rank (↑ = higher rank)',
        'Overall Rank (↑ = higher rank)'  # ← new column label
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
plt.savefig(os.path.join(graphs_path, "streaks_and_switches_summary.png"), bbox_inches='tight')
plt.show()

###############################################################################
###### Streaks and switching median

# Function: calculate median streak and switches per treatment
def summarize_all_median(data):
    # Streaks
    streaks = data.groupby(['Participant id'])['streak_length'].median().reset_index(name='median_streak')
    # Switches CRRA
    switches_crra = data.loc[data['crra_diff'].notna() & (data['crra_diff'] != 0)]
    switches_crra_summary = switches_crra.groupby('Participant id')['crra_diff'].median().reset_index(name='median_switch_crra')
    # Switches displayed rank
    switches_rank = data.loc[data['rank_diff'].notna() & (data['rank_diff'] != 0)]
    switches_rank_summary = switches_rank.groupby('Participant id')['rank_diff'].median().reset_index(name='median_switch_rank')
    # Switches overall rank
    switches_overallrank = data.loc[data['overallRank_diff'].notna() & (data['overallRank_diff'] != 0)]
    switches_overallrank_summary = switches_overallrank.groupby('Participant id')['overallRank_diff'].median().reset_index(name='median_switch_overallrank')

    # Merge treatment info
    treatments = data.drop_duplicates('Participant id')[['Participant id', 'TreatmentLabel']]
    streaks = streaks.merge(treatments, on='Participant id')
    switches_crra_summary = switches_crra_summary.merge(treatments, on='Participant id')
    switches_rank_summary = switches_rank_summary.merge(treatments, on='Participant id')
    switches_overallrank_summary = switches_overallrank_summary.merge(treatments, on='Participant id')

    # Group by treatment
    median_streaks = streaks.groupby('TreatmentLabel')['median_streak'].median().round(2)
    median_switches_crra = switches_crra_summary.groupby('TreatmentLabel')['median_switch_crra'].median().round(2)
    median_switches_rank = switches_rank_summary.groupby('TreatmentLabel')['median_switch_rank'].median().round(2)
    median_switches_overallrank = switches_overallrank_summary.groupby('TreatmentLabel')['median_switch_overallrank'].median().round(2)

    # Combine into single table
    combined = pd.concat([median_streaks, median_switches_crra, median_switches_rank, median_switches_overallrank], axis=1)
    combined.columns = [
        'Median Streak Length',
        'Median Switch CRRA (↑ = less risk)',
        'Rank (↑ = higher rank)',
        'Overall Rank (↑ = higher rank)'
    ]

    return combined

# Collect median results
median_results = {}
median_results['Combined'] = summarize_all_median(df)

for path in sorted(df['pathseries'].unique()):
    median_results[f'Pathseries {path}'] = summarize_all_median(df[df['pathseries'] == path])

# Plotting medians as tables in one figure
fig, axs = plt.subplots(len(median_results), 1, figsize=(12, 3.5 * len(median_results)))
if len(median_results) == 1:
    axs = [axs]

for ax, (title, table_df) in zip(axs, median_results.items()):
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
    ax.set_title(f'{title} (Median)', fontweight='bold', pad=10)

plt.tight_layout()

# Save and show median plot
plt.savefig(os.path.join(graphs_path, "streaks_and_switches_summary_median.png"), bbox_inches='tight')
plt.show()

###############################################################################
###### Graphs
###############################################################################

### cumulative rank over phases
# Create treatment labels
df = data.copy()
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
plt.suptitle('Cumulative Rank Over Phases', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "avg_cumulative_rank.png"), bbox_inches='tight')
plt.show()

##############################################################################
### cumulative rank over phases
# Create treatment labels
df = panel_df_clean_bot.copy()
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Ensure sorting
df = df.sort_values(by=['Participant id', 'phase'])

# Compute cumulative copiedRank per participant
df['cumulativeRank'] = df.groupby('Participant id')['overallRank'].cumsum()

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
    ax.set_ylabel('Avg Cumulative overall Rank')
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined plot
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in cumulative_combined.columns:
    ax_combined.plot(cumulative_combined.index, cumulative_combined[col], label=col)
ax_combined.set_title('Combined Pathseries')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Avg Cumulative overall Rank')
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')
plt.suptitle('Cumulative Overall Rank Over Phases', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "avg_cumulative_overall_rank.png"), bbox_inches='tight')
plt.show()

###############################################################################
### cumulative risk rank over phases
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
plt.suptitle('Cumulative Risk Rank Over Phases', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "avg_cumulative_risk_rank.png"), bbox_inches='tight')
plt.show()

###############################################################################
### cumulative overall rank over phases
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Ensure sorting
df = df.sort_values(by=['Participant id', 'phase'])

# Compute cumulative copiedRank per participant
df['cumulativeRank'] = df.groupby('Participant id')['overallRank'].cumsum()

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
    ax.set_ylabel('Avg cumulative copied overall rank')
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
ax_combined.set_ylabel('Avg cumulative copied overall rank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')
plt.suptitle('Cumulative Overall Rank Over Phases', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "avg_cumulative_overall_rank.png"), bbox_inches='tight')
plt.show()

###############################################################################
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
plt.savefig(os.path.join(graphs_path, "smoothed_copiedRank_vs_copiedCRRA_rank.png"), bbox_inches='tight')
plt.show()
plt.show()

###############################################################################
#### copied rank vs copied overall rank smoothed
df = panel_df_clean_bot.copy()
df = df.sort_values(by=['Participant id', 'phase'])

# Create treatment labels
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Compute cumulative ranks per participant
df['cumulative_copiedRank'] = df.groupby('Participant id')['copiedRank'].cumsum()
df['cumulative_overall_rank'] = df.groupby('Participant id')['overallRank'].cumsum()

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
    cumulative_crra_dfs[path] = average_cumulative_by_phase(df[df['pathseries'] == path], 'cumulative_overall_rank')
cumulative_combined_rank = average_cumulative_by_phase(df, 'cumulative_copiedRank')
cumulative_combined_crra = average_cumulative_by_phase(df, 'cumulative_overall_rank')

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
    ax.set_title(f'Cumulative Displayed vs Overall Rank (Pathseries {path})')
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
ax_combined.set_title('Cumulative Displayed vs overall Rank (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Avg Cumulative Value')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "smoothed_copiedRank_vs_overall_rank.png"), bbox_inches='tight')
plt.show()

###############################################################################
### Average number of change of copied bot
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
plt.savefig(os.path.join(graphs_path, "cumulative_average_changes.png"), bbox_inches='tight')
plt.show()

###############################################################################
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
plt.savefig(os.path.join(graphs_path, "risk_trajectory.png"), bbox_inches='tight')
plt.show()

###############################################################################
#### Risk trajectory paths median
df = panel_df_clean.copy()
df['copiedCRRA_rank'] = pd.to_numeric(df['copiedCRRA_rank'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedCRRA_rank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['copiedCRRA_rank'].median().reset_index()
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
plt.savefig(os.path.join(graphs_path, "risk_trajectory_median.png"), bbox_inches='tight')
plt.show()

###############################################################################
#### Rank trajectory paths - AVG
df = panel_df_clean_bot.copy()
df['overallRank'] = pd.to_numeric(df['overallRank'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedCRRA_rank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['overallRank'].mean().reset_index()
    pivot = grouped.pivot(index='phase', columns='TreatmentLabel', values='overallRank')
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
    ax.set_title(f'Overall Rank Trajectory (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average overall rank')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_combined.columns:
    ax_combined.plot(average_combined.index, average_combined[col], label=col, marker='o')
ax_combined.set_title('Overall Rank Trajectory (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Average overall rank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')
# Add main title
plt.suptitle('Average Overall Rank Over Phases per Treatment', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "overall_rank_trajectory.png"), bbox_inches='tight')
plt.show()

###############################################################################
#### Rank trajectory paths - MEDIAN
df = panel_df_clean_bot.copy()
df['overallRank'] = pd.to_numeric(df['overallRank'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedCRRA_rank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['overallRank'].median().reset_index()
    pivot = grouped.pivot(index='phase', columns='TreatmentLabel', values='overallRank')
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
    ax.set_title(f'Overall Rank Trajectory (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average overall rank')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_combined.columns:
    ax_combined.plot(average_combined.index, average_combined[col], label=col, marker='o')
ax_combined.set_title('Overall Rank Trajectory (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Average overall rank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')
# Add main title
plt.suptitle('Median Overall Rank Over Phases per Treatment', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "overall_rank_trajectory_median.png"), bbox_inches='tight')
plt.show()

###############################################################################
#### Rank trajectory paths - AVG
df = panel_df_clean_bot.copy()
df['copiedRank'] = pd.to_numeric(df['copiedRank'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedRank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['copiedRank'].mean().reset_index()
    pivot = grouped.pivot(index='phase', columns='TreatmentLabel', values='copiedRank')
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
    ax.set_title(f'Overall Rank Trajectory (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average rank')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_combined.columns:
    ax_combined.plot(average_combined.index, average_combined[col], label=col, marker='o')
ax_combined.set_title('Overall Rank Trajectory (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Average rank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')
# Add main title
plt.suptitle('Average Rank Over Phases per Treatment', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "rank_trajectory.png"), bbox_inches='tight')
plt.show()

###############################################################################
### Rank trajectory only for those that change
df = panel_df_clean_bot.copy()
df = df[df['copiedCRRA_change'] == 1]
df['copiedRank'] = pd.to_numeric(df['copiedRank'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedRank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['copiedRank'].mean().reset_index()
    pivot = grouped.pivot(index='phase', columns='TreatmentLabel', values='copiedRank')
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
    ax.set_title(f'Copied Rank Trajectory (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average rank')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_combined.columns:
    ax_combined.plot(average_combined.index, average_combined[col], label=col, marker='o')
ax_combined.set_title('Copied Rank Trajectory (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Average rank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

# Add main title
plt.suptitle('Average Rank Copied Over Phases per Treatment\n(Only when copying choice changed)', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "rank_trajectory_copiedCRRA_change1.png"), bbox_inches='tight')
plt.show()

###############################################################################
### Rank trajectory only for those that change
df = panel_df_clean_bot.copy()
df = df[df['copiedCRRA_change'] == 1]
df['overallRank'] = pd.to_numeric(df['overallRank'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedRank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['overallRank'].mean().reset_index()
    pivot = grouped.pivot(index='phase', columns='TreatmentLabel', values='overallRank')
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
    ax.set_title(f'Overall Rank Trajectory (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average rank')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_combined.columns:
    ax_combined.plot(average_combined.index, average_combined[col], label=col, marker='o')
ax_combined.set_title('Overall Rank Trajectory (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Average rank')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')

# Add main title
plt.suptitle('Average Overall Rank Copied Over Phases per Treatment\n(Only when copying choice changed)', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "overall_rank_trajectory_copiedCRRA_change1.png"), bbox_inches='tight')
plt.show()

###############################################################################
#### Risk deviation trajectory paths - lottery
df = panel_df_clean_bot.copy()
df['CRRAchange'] = pd.to_numeric(df['CRRAchange'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedCRRA_rank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['CRRAchange'].mean().reset_index()
    pivot = grouped.pivot(index='phase', columns='TreatmentLabel', values='CRRAchange')
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
    ax.set_title(f'Risk Deviation Trajectory (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average CRRAchange')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_combined.columns:
    ax_combined.plot(average_combined.index, average_combined[col], label=col, marker='o')
ax_combined.set_title('Risk Deviation Trajectory (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Average CRRAchange')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')
plt.suptitle('Average Risk Deviation with Lottery Choice', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "risk_dediation_trajectory.png"), bbox_inches='tight')
plt.show()

###############################################################################
#### Risk deviation trajectory paths - Falk
df = panel_df_clean_bot.copy()
df['Falkchange'] = pd.to_numeric(df['Falkchange'], errors='coerce')

# Create TreatmentGroup label
name_map = {1: 'R', 0: 'NR'}
followers_map = {1: 'F', 0: 'NF'}
df['TreatmentLabel'] = df['nameTreatment'].map(name_map) + '+' + df['followersTreatment'].map(followers_map)

# Function to compute average copiedCRRA_rank per phase and treatment
def average_rank_by_phase(data):
    grouped = data.groupby(['phase', 'TreatmentLabel'])['Falkchange'].mean().reset_index()
    pivot = grouped.pivot(index='phase', columns='TreatmentLabel', values='Falkchange')
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
    ax.set_title(f'Risk Deviation Trajectory (Pathseries {path})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average Falk change')
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if idx == 1:
        ax.legend(title='Treatment')

# Second row: combined
ax_combined = plt.subplot2grid((2, 2), (1, 0), colspan=2)
for col in average_combined.columns:
    ax_combined.plot(average_combined.index, average_combined[col], label=col, marker='o')
ax_combined.set_title('Risk Deviation Trajectory (Combined Pathseries)')
ax_combined.set_xlabel('Phase')
ax_combined.set_ylabel('Average Falk change')
ax_combined.set_ylim(ymin, ymax)
ax_combined.grid(True)
ax_combined.legend(title='Treatment', loc='upper left')
plt.suptitle('Average Risk Deviation with Falk catergory', fontsize=16, y=1.02)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(graphs_path, "risk_dediation_trajectory_falk.png"), bbox_inches='tight')
plt.show()

###############################################################################
### Scatter CRRA Lottery vs Falk

df = panel_df_clean_bot[panel_df_clean_bot['phase'] == 0].copy()
# Add jittering
np.random.seed(0)  # for reproducibility
jitter_strength = 0.1
x_jittered = df['CRRA_rank'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
y_jittered = df['RiskFalk_cat'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))

plt.figure(figsize=(8, 6))

# Scatter plot with jitter
plt.scatter(x_jittered, y_jittered, alpha=0.6, label='Data points')

# Regression line using seaborn
sns.regplot(x='CRRA_rank', y='RiskFalk_cat', data=df, scatter=False, ci=None, color='red', label='Regression line')

# Labels and title
plt.xlabel('CRRA_rank (1 = highest risk, 5 = lowest risk)')
plt.ylabel('RiskFalk_cat (1 = highest risk, 5 = lowest risk)')
plt.title('Scatter plot of CRRA_rank vs. RiskFalk_cat with jitter and regression')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(graphs_path, "lottery_vs_falk.png"), bbox_inches='tight')
plt.show()

###############################################################################
#### Cross Tabulation
# Create count and proportion crosstabs
counts = pd.crosstab(df['CRRA_rank'], df['RiskFalk_cat'])
props = pd.crosstab(df['CRRA_rank'], df['RiskFalk_cat'], normalize='all')

# Combine into single formatted table: count (prop)
formatted_table = counts.astype(str) + " (" + props.round(3).astype(str) + ")"

# Plot the table using matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Add the table
table = ax.table(
    cellText=formatted_table.values,
    rowLabels=[f'CRRA_rank {i}' for i in formatted_table.index],
    colLabels=[f'RiskFalk_cat {i}' for i in formatted_table.columns],
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Add overall title and axis labels
plt.title('Cross-tabulation of CRRA_rank vs RiskFalk_cat\nCounts with proportions (total)', fontsize=12, pad=20)


# Save to file
plt.savefig(os.path.join(graphs_path, "crosstab_lottery_vs_falk.png"), bbox_inches='tight')
plt.show()


###### Actual testing
from scipy.stats import mannwhitneyu
from statsmodels.stats.power import TTestIndPower
#### increased risk taking
df_phase0 = panel_df_clean_bot[panel_df_clean_bot['phase'] == 0]


group1 = df_phase0['CRRA_rank']
group2 = df_phase0['copiedCRRA_rank']

stat, p = mannwhitneyu(group1, group2, alternative='greater')  
print("Pooled path series")
print(f"U-statistic: {stat}")
print(f"P-value: {p}")

###Calculate power
n1 = len(group1)
n2 = len(group2)
r = 1 - (2 * stat) / (n1 * n2)
print(f"Rank-biserial correlation: {r}")
analysis = TTestIndPower()
required_n = analysis.solve_power(effect_size=abs(r), power=0.8, alpha=0.05, alternative='larger')
print(f"Required sample size per group: {required_n:.2f}")

#### series 6 increased risk taking
df_phase0 = panel_df_clean_bot[panel_df_clean_bot['phase'] == 0]
df_phase0 = df_phase0[df_phase0['pathseries'] == 6]

group1 = df_phase0['CRRA_rank']
group2 = df_phase0['copiedCRRA_rank']

stat, p = mannwhitneyu(group1, group2, alternative='greater')  
print("Pathseries 6")
print(f"U-statistic: {stat}")
print(f"P-value: {p}")

#### series 2 increased risk taking
df_phase0 = panel_df_clean_bot[panel_df_clean_bot['phase'] == 0]
df_phase0 = df_phase0[df_phase0['pathseries'] == 2]

group1 = df_phase0['CRRA_rank']
group2 = df_phase0['copiedCRRA_rank']
stat, p = mannwhitneyu(group1, group2, alternative='greater')  
print("Pathseries 2")
print(f"U-statistic: {stat}")
print(f"P-value: {p}")

#### Treatments comparison
#### Names
df_phase0 = panel_df_clean_bot[panel_df_clean_bot['phase'] == 0]

group1 = df_phase0[df_phase0['nameTreatment'] == 1]['copiedCRRA_rank']
group2 = df_phase0[df_phase0['nameTreatment'] == 0]['copiedCRRA_rank']
stat, p = mannwhitneyu(group1, group2, alternative='two-sided') 
print("Risk names vs generic names")
print(f"U-statistic: {stat}")
print(f"P-value: {p}")

#### Followers
df_phase0 = panel_df_clean_bot[panel_df_clean_bot['phase'] == 0]

group1 = df_phase0[df_phase0['followersTreatment'] == 1]['copiedCRRA_rank']
group2 = df_phase0[df_phase0['followersTreatment'] == 0]['copiedCRRA_rank']
stat, p = mannwhitneyu(group1, group2, alternative='two-sided') 
print("Followers vs no followers")
print(f"U-statistic: {stat}")
print(f"P-value: {p}")

###Calculate power
n1 = len(group1)
n2 = len(group2)
r = 1 - (2 * stat) / (n1 * n2)
print(f"Rank-biserial correlation: {r}")
analysis = TTestIndPower()
required_n = analysis.solve_power(effect_size=abs(r), power=0.8, alpha=0.05, alternative='larger')
print(f"Required sample size per group: {required_n:.2f}")