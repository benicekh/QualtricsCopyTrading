import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

# Folder where the files are stored
folder_paths = [
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Botdata",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game\Botdata"
]

#Automatically select the first existing folder
folder_path = next((path for path in folder_paths if os.path.exists(path)), None)

if folder_path is None:
    raise FileNotFoundError("None of the specified folder paths exist.")

print(f"Using folder path: {folder_path}")

# Dictionary to hold all series
all_series_data = {}

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
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

print(all_series_data.keys())

import json

with open('data.json', 'w') as f:
    json.dump(all_series_data["series 2"], f)

data = all_series_data["series 2"].copy()
    
# Prepare the table
table = {}
phase_labels = ['Start'] + [f'Phase {i+2}' for i in range(9)]  # Start, Phase 2, ..., Phase 10

for stage_idx in range(10):
    stage_key = f'stage_{stage_idx}'
    round_key = 'round_0'
    phase_label = 'Start' if stage_idx == 0 else f'Phase {stage_idx + 1}'
    
    row = []
    for rank in range(1, 6):  # ranks 1–5
        try:
            rank_data = data[stage_key][round_key][f'rank_{rank}']
            cell = f"{rank_data['ResponseId']} / {rank_data['phaseReturn']} / {rank_data['returnAllv2']}"
        except KeyError:
            cell = 'NA'
        row.append(cell)
    
    table[phase_label] = row

# Create DataFrame
df = pd.DataFrame.from_dict(table, orient='index', columns=[f'Rank {i}' for i in range(1, 6)])

# Plot DataFrame as a table
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')
table_plot = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

# Adjust table style
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(10)
table_plot.scale(1.2, 1.2)

# Add title
plt.title('Pathseries 2', fontsize=16, pad=20)

# Add legend as text below the table
plt.figtext(0.5, 0.01, 'Legend: Bot ID / Last Phase Return / Overall Return', ha='center', fontsize=10)

# Save as PNG
plt.savefig('phase_rank_table.png', bbox_inches='tight')
#plt.close()

print("Saved table as 'phase_rank_table.png'")

############# Series 6

data = all_series_data["series 6"].copy()
    
# Prepare the table
table = {}
phase_labels = ['Start'] + [f'Phase {i+2}' for i in range(9)]  # Start, Phase 2, ..., Phase 10

for stage_idx in range(10):
    stage_key = f'stage_{stage_idx}'
    round_key = 'round_0'
    phase_label = 'Start' if stage_idx == 0 else f'Phase {stage_idx + 1}'
    
    row = []
    for rank in range(1, 6):  # ranks 1–5
        try:
            rank_data = data[stage_key][round_key][f'rank_{rank}']
            cell = f"{rank_data['ResponseId']} / {rank_data['phaseReturn']} / {rank_data['returnAllv2']}"
        except KeyError:
            cell = 'NA'
        row.append(cell)
    
    table[phase_label] = row

# Create DataFrame
df = pd.DataFrame.from_dict(table, orient='index', columns=[f'Rank {i}' for i in range(1, 6)])

# Plot DataFrame as a table
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')
table_plot = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

# Adjust table style
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(10)
table_plot.scale(1.2, 1.2)

# Add title
plt.title('Pathseries 6', fontsize=16, pad=20)

# Add legend as text below the table
plt.figtext(0.5, 0.01, 'Legend: Bot ID / Last Phase Return / Overall Return', ha='center', fontsize=10)

# Save as PNG
plt.savefig('phase_rank_table.png', bbox_inches='tight')
#plt.close()

print("Saved table as 'phase_rank_table.png'")

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

# Optional: sort nicely
botdata_df = botdata_df.sort_values(by=['pathseries', 'phase', 'overallRank']).reset_index(drop=True)

# Show final DataFrame
print(botdata_df)

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

# Merge
merged_df = panel_df_clean.merge(
    botdata_filtered,
    how='left',
    on=['tradeLeader', 'phase', 'pathseries']
)

merged_df = merged_df.sort_values(by=['Submission id', 'phase', 'rank']).reset_index(drop=True)
panel_df_clean_bot = merged_df.sort_values(by=['SubmissionId', 'phase', 'rank']).reset_index(drop=True)