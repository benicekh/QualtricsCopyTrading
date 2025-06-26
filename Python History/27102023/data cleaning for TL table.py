# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:27:55 2023

@author: benja
"""

import pandas as pd
import os
import json
import re

os.chdir(r"C:\Users\U338144\surfdrive\Projects\Copy Trading\Qualtrics Exp\Python - Ordered List")
df = pd.read_csv("pilot2310partial.csv")
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

for i in range(2, len(df)):
    variable_name = "player_" + str(i-1)
    temp = json.loads(df.Data[i])
    data_players[variable_name] = pd.DataFrame.from_dict(temp['rounds'])
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
    for k in range(83, 85):
        varname = f"{df.iloc[1, k]}"
        # Extract text within the second set of quotation marks
        match = re.search(r'{"[^"]*":"([^"]*)"}', varname)
        if match:
            extracted_text = match.group(1)
            varname = f"{extracted_text}"
        data_players[variable_name][varname] = df.iloc[i, k]
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
    column_to_drop = 'treatment_y'
    if column_to_drop in data_players[variable_name].columns:
        data_players[variable_name].drop(column_to_drop, axis=1, inplace=True)
    column_to_drop = 'treatment_x'
    if column_to_drop in data_players[variable_name].columns:
        data_players[variable_name].drop(column_to_drop, axis=1, inplace=True)
                                     
# Define the output folder
output_folder = 'csv'
                                         
for key, df in data_players.items():
    csv_filename = f"{output_folder}/{key}.csv"  # Define the full path to the CSV file
    df.to_csv(csv_filename, index=False)

data_rounds = {}

# Sorting/Ranking 0 - ongoingReturn, 1 - Rank, 2 - ResponseID, 3 - gainAll, 4 - returnAll
for i in range(0, data_players["player_1"]["stg"].nunique()) :
    stage = "stage_" + str(i)
    data_rounds[stage] = {}
    for n in range(0, 20):
          rnd = "round_" + str(n)
          data_rounds[stage][rnd] = {}
          for j in range(0, len(data_players)):
              player = "player_" + str(j+1)
              data_rounds[stage][rnd][player] = data_players[player].loc[(data_players[player]["stg"] == i) & \
              (data_players[player]["r"] == n) & (data_players[player]["phase"] == "regular")]
          for key in  data_rounds[stage][rnd].keys():
              for other_key in  data_rounds[stage][rnd].keys():
                  if float(data_rounds[stage][rnd][other_key].iloc[0, 3]) > float(data_rounds[stage][rnd][key].iloc[0, 3]):
                      data_rounds[stage][rnd][key].iloc[0,1] += 1  
          for m in range (1, len(data_players)):
              for key in  data_rounds[stage][rnd].keys():
                  for other_key in  data_rounds[stage][rnd].keys():
                      if data_rounds[stage][rnd][other_key].iloc[0, 1] == data_rounds[stage][rnd][key].iloc[0, 1] \
                      and data_rounds[stage][rnd][other_key].iloc[0, 2] != data_rounds[stage][rnd][key].iloc[0, 2]:
                          data_rounds[stage][rnd][key].iloc[0,1] += 1  


data_ranked = {}

for i in range(0, data_players["player_1"]["stg"].nunique()) :
    stage = "stage_" + str(i)
    data_ranked[stage] = {}
    for n in range(0, 20):
        rnd = "round_" + str(n)
        data_ranked[stage][rnd] = {}
        for j in range(0, len(data_players)):
            rank = "rank_" + str(j+1)
            data_ranked[stage][rnd][rank] = ()
            for key in  data_rounds[stage][rnd].keys():
                if data_rounds[stage][rnd][key].iloc[0,1] == j+1:
                    data_ranked[stage][rnd][rank] = data_rounds[stage][rnd][key]     
                    

dfOutput = pd.DataFrame()
for i in range(0, data_players["player_1"]["stg"].nunique()) :
    stage = "stage_" + str(i)
    for n in range(0, 20):
        rnd = "round_" + str(n)
        for key in  data_rounds[stage][rnd].keys():
            dfOutput = dfOutput.append(data_rounds[stage][rnd][key], ignore_index=True)
            
dfOutput.to_csv( "pilotJSdata2310partial.csv", index=False, encoding='utf-8-sig')

string = "const data = {"
string += "\n"

for i in range(1, data_players["player_1"]["stg"].nunique()) :
    stage = "stage_" + str(i)
    string += "\t"
    string += f'{stage}: '
    string += '{'
    string += "\n"
    for n in range(0, 20):
        rnd = "round_" + str(n)
        string += "\t\t"
        string += f'{rnd}: '
        string += '{'
        string += "\n"
        for j in range(0, len(data_players)):
            rank = "rank_" + str(j+1)
            string += "\t\t\t"
            string += f'{rank}: '
            string += '{'
            string += "\n"
            for key in data_ranked[stage][rnd][rank].keys():
                string += "\t\t\t\t"
                if key == "phase" or key == "ResponseId" or key == "path":
                    string += str(f'{key}:"{data_ranked[stage][rnd][rank][key].iloc[0]}",')
                    string += "\n"
                else:
                    string += str(f'{key}:{data_ranked[stage][rnd][rank][key].iloc[0]},')
                    string += "\n"
            string += "\t\t\t\t"
            string += "},"
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


file_name = "stringPilot2310partial.txt"        
export_string_to_txt_file(string, file_name)
