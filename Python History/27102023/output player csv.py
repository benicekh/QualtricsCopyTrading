# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:57:04 2023

@author: U338144
"""


import pandas as pd
import os
import json
import re

os.chdir(r"C:\Users\U338144\surfdrive\Projects\Copy Trading\Qualtrics Exp\Python - Ordered List")
df = pd.read_csv("pilot1710.csv")
df.dropna(subset=['Data'], inplace=True)
df = df.reset_index(drop=True)

data_players_stagesummaries = {}

for i in range(2, len(df)):
    variable_name = "player_" + str(i-1)
    temp = json.loads(df.Data[i])
    data_players_stagesummaries[variable_name] = pd.DataFrame.from_dict(temp['stagesummaries'])
    data_players_stagesummaries[variable_name] = data_players_stagesummaries[variable_name].drop(0)
    for index, value in data_players_stagesummaries[variable_name]["ongoingReturn"].iteritems():
        print(f'Index: {index}, Value: {value}')
        integer_list = [float(x) for x in data_players_stagesummaries[variable_name]["ongoingReturn"][index]]
        data_players_stagesummaries[variable_name]["ongoingReturn"][index] = integer_list
    data_players_stagesummaries[variable_name]["ongoingReturnSeries"]=data_players_stagesummaries[variable_name]["ongoingReturn"]   
    data_players_stagesummaries[variable_name]["phaseWealth"] = data_players_stagesummaries[variable_name]["wealth"]
    data_players_stagesummaries[variable_name]["stg"] = data_players_stagesummaries[variable_name]["phase"]+1
    data_players_stagesummaries[variable_name]["ResponseId"] = df.loc[i, "ResponseId"]
    data_players_stagesummaries[variable_name] = data_players_stagesummaries[variable_name].drop(columns="ongoingReturn", axis=1)

data_players_stagesummaries_series = {}
for i in range(2, len(df)):
    variable_name = "player_" + str(i-1)
    temp = json.loads(df.Data[i])
    data_players_stagesummaries_series[variable_name] = pd.DataFrame.from_dict(temp['stagesummaries'])
    data_players_stagesummaries_series[variable_name] = data_players_stagesummaries_series[variable_name].drop(0)
    data_players_stagesummaries_series[variable_name]["stg"] = data_players_stagesummaries[variable_name]["phase"]+1
    columns_to_drop = ['phaseName', 'phase','wealth', 'gain', 'phaseReturn', 'wealthALL', 'gainAll', 'returnAll', 'tradeLeader',  'treatment'  ]
    data_players_stagesummaries_series[variable_name] = data_players_stagesummaries_series[variable_name].drop(columns=columns_to_drop, axis=1)


data_players = {}

print(df.iloc[1, 61])
for k in range(10, 90):
    print(k)
    print(df.iloc[1, k])

for i in range(2, len(df)):
    variable_name = "player_" + str(i-1)
    temp = json.loads(df.Data[i])
    data_players[variable_name] = pd.DataFrame.from_dict(temp['rounds'])
    varname = f"{df.iloc[0, 4]}"
    data_players[variable_name][varname] = df.iloc[i, 4]
    varname = f"{df.iloc[0, 6]}"
    data_players[variable_name][varname] = df.iloc[i, 6]
    
    for k in range(10, 85):
        varname = f"{df.iloc[1, k]}"
        # Extract text within the second set of quotation marks
        match = re.search(r'{"[^"]*":"([^"]*)"}', varname)
        if match:
            extracted_text = match.group(1)
            varname = f"{extracted_text}"
        data_players[variable_name][varname] = df.iloc[i, k]
    
    
    data_players[variable_name]["ResponseId"] = df.loc[i, "ResponseId"]
    data_players[variable_name]["PlayerName"] = f'"{variable_name}"'
    data_players[variable_name]["Rank"] = 1
    data_players[variable_name] = data_players[variable_name].merge(data_players_stagesummaries[variable_name], on='stg', how='left')
    data_players[variable_name]["phase"] = data_players[variable_name]["phase_x"]
    data_players[variable_name]["ResponseId"] = data_players[variable_name]["ResponseId_x"]
    columns_to_drop = ['phase_x', 'ResponseId_x','ResponseId_y', "phase_y", "wealth", "clicks", "next", "previous", "tradeLeader", "phaseName" ]
    data_players[variable_name] = data_players[variable_name].drop(columns=columns_to_drop, axis=1)
    data_players[variable_name]["GainCalc"] = data_players[variable_name]["c"] - (625 - data_players[variable_name]["unrealized"])
    data_players[variable_name]["GainCalc"] = round(data_players[variable_name]["GainCalc"], 2)
    data_players[variable_name]["realizedReturnCalc"] = (((625+data_players[variable_name]["GainCalc"])/625)-1)*100
    data_players[variable_name]["realizedReturnCalc"] = round(data_players[variable_name]["realizedReturnCalc"], 2)
    columns_to_move = ["ongoingReturn", "Rank", "ResponseId", "gainAll", "returnAll"] 
    new_order = columns_to_move + [col for col in  data_players[variable_name].columns if col not in columns_to_move]
    data_players[variable_name] =  data_players[variable_name][new_order]
    for j in range(125, 145):
        roundn = j-125
        data_players[variable_name].loc[j] = data_players[variable_name].loc[124]
        data_players[variable_name]["r"][j] = roundn
        data_players[variable_name]["stg"][j] = 6
        data_players[variable_name]["gain"][j] = (data_players[variable_name]["p"][124]*data_players[variable_name]["a"][124])+data_players[variable_name]["c"][124]-625
        data_players[variable_name]["gainAll"][j] = data_players[variable_name]["gainAll"][124]+data_players[variable_name]["gain"][j]
        data_players[variable_name]["wealthALL"][j] = 6*625 + data_players[variable_name]["gainAll"][j]
        data_players[variable_name]["phaseWealth"][j] = data_players[variable_name]["gain"][j]+625
        data_players[variable_name]["phaseReturn"][j] = data_players[variable_name]["ongoingReturn"][124]
        data_players[variable_name]["returnAll"][j] = round((data_players[variable_name]["wealthALL"][j]/(6*625)-1)*100, 2)
        data_players[variable_name]["roundSeries"][j] = data_players_stagesummaries_series[variable_name]['roundSeries'][6]
        data_players[variable_name]["priceSeries"][j] = data_players_stagesummaries_series[variable_name]['priceSeries'][6]
        data_players[variable_name]["assetsSeries"][j]= data_players_stagesummaries_series[variable_name]['assetsSeries'][6]
        data_players[variable_name]["ongoingReturnSeries"][j]= data_players_stagesummaries_series[variable_name]['ongoingReturn'][6]

                                     
# Define the output folder
output_folder = 'csv'
                                         
for key, df in data_players.items():
    csv_filename = f"{output_folder}/{key}.csv"  # Define the full path to the CSV file
    df.to_csv(csv_filename, index=False)