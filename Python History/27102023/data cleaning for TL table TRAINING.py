# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:27:55 2023

@author: benja
"""

import pandas as pd
import os
import json

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
    assetSeries = "["
    priceSeries = "["
    ongoingReturnSeries = "["
    for j in range(0, 5):
        contentAsset = str(data_players[variable_name]["a"][j])
        contentPrice = str(data_players[variable_name]["p"][j])
        contentReturn = str(data_players[variable_name]["ongoingReturn"][j])
        assetSeries += f"{contentAsset}"
        priceSeries += f"{contentPrice}"
        ongoingReturnSeries += f"{contentReturn}"
        if j != 4:
            assetSeries += ","
            priceSeries += ","
            ongoingReturnSeries += ","
    assetSeries += "]"
    priceSeries += "]"
    ongoingReturnSeries += "]"
    roundSeries = "[0,1,2,3,4]"
    for k in range(0, 5):
        data_players[variable_name]["roundSeries"][k] = roundSeries
        data_players[variable_name]["priceSeries"][k] = priceSeries
        data_players[variable_name]["assetsSeries"][k] = assetSeries
        data_players[variable_name]["ongoingReturnSeries"][k] = ongoingReturnSeries
        data_players[variable_name]["phaseWealth"][k] =  data_players[variable_name]["a"][4]*data_players[variable_name]["p"][4]+data_players[variable_name]["c"][4]
        data_players[variable_name]["wealthALL"][k] = data_players[variable_name]["phaseWealth"][k]     
        data_players[variable_name]["phaseReturn"][k] = data_players[variable_name]["ongoingReturn"][4] 
        data_players[variable_name]["returnAll"][k] = data_players[variable_name]["ongoingReturn"][4] 
        data_players[variable_name]["gain"][k] = data_players[variable_name]["c"][4]-625
        data_players[variable_name]["gainAll"][k] = data_players[variable_name]["c"][4]-625                                           


data_rounds = {}

# Sorting/Ranking 0 - ongoingReturn, 1 - Rank, 2 - ResponseID, 3 - gainAll, 4 - returnAll
stage = "stage_" + str(0)
data_rounds[stage] = {}
for n in range(0, 5):
      rnd = "round_" + str(n)
      data_rounds[stage][rnd] = {}
      for j in range(0, len(data_players)):
          player = "player_" + str(j+1)
          data_rounds[stage][rnd][player] = data_players[player].loc[(data_players[player]["stg"] == 0) & \
          (data_players[player]["r"] == n) & (data_players[player]["phase"] == "training")]
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


stage = "stage_" + str(0)
data_ranked[stage] = {}
for n in range(0, 5):
    rnd = "round_" + str(n)
    data_ranked[stage][rnd] = {}
    for j in range(0, len(data_players)):
        rank = "rank_" + str(j+1)
        data_ranked[stage][rnd][rank] = ()
        for key in  data_rounds[stage][rnd].keys():
            if data_rounds[stage][rnd][key].iloc[0,1] == j+1:
                data_ranked[stage][rnd][rank] = data_rounds[stage][rnd][key]     
                    

dfOutput = pd.DataFrame()

stage = "stage_" + str(0)
for n in range(0, 5):
    rnd = "round_" + str(n)
    for key in  data_rounds[stage][rnd].keys():
        dfOutput = dfOutput.append(data_rounds[stage][rnd][key], ignore_index=True)
            
dfOutput.to_csv( "pilotJSdata.csv", index=False, encoding='utf-8-sig')

string = "const dataTraining = {"
string += "\n"


stage = "stage_" + str(0)
string += "\t"
string += f'{stage}: '
string += '{'
string += "\n"
for n in range(0, 5):
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


file_name = "stringPilotTraining24101.txt"        
export_string_to_txt_file(string, file_name)
