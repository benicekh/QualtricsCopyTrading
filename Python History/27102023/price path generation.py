# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:13:41 2023

@author: U338144
"""

from random import *
import pandas as pd
import os

os.chdir("D:\\Surfdrive\\Projects\\Copy Trading\\Qualtrics Exp\\Copy-trading-paper")


data = {}

occurence = []
for n in range (0, 1):      
    for j in range(0,100000):
        data[j] = [125]
        good = False
        x = random() * 100
        if x <= 50:
            good = True
        for i in range(0,39):
            if good == True:
                y = random() * 100
                if y <= 75:
                    x = random()
                    if x <= (1/3):
                        p = data[j][i] + 2
                    elif x > (2/3):
                        p = data[j][i] + 4
                    else:
                        p = data[j][i] + 8
                else:
                    x = random()
                    if x <= (1/3):
                        p = data[j][i] - 2
                    elif x > (2/3):
                        p = data[j][i] - 4
                    else:
                        p = data[j][i] - 8
                z = random() * 100
                if z <= 15:
                    good = False
                            
            else:
                y = random() * 100
                if y <= 75:
                    x = random()
                    if x <= (1/3):
                        p = data[j][i] - 2
                    elif x > (2/3):
                        p = data[j][i] - 4
                    else:
                        p = data[j][i] - 8
                else:
                    x = random()
                    if x <= (1/3):
                        p = data[j][i] + 2
                    elif x > (2/3):
                        p = data[j][i] + 4
                    else:
                        p = data[j][i] + 8
                z = random() * 100
                if z <= 15:
                    good = True
            if p >= 0:
                data[j].append(p)
            else:
                data[j].append(0)
    
    count = 0
    for j in range(0,100000):
        temp = data[j].count(0)
        if temp > 0:
            count += 1
    occurence.append(count)
    print(n)    

choiceSet = {}

for j in range(0,100):
    x = int(random() * 100000)

    choiceSet[j] = data[x]

string = "["
string += "\n\t{"
string += "\n\t\t\"period\":"
string += "\n\""

for j in range(0, len(choiceSet[1])):
    string += str(j)
    for i in range(0, len(choiceSet)):
        string += ","
        string += str(f'{choiceSet[i][j]}')
    string += ";"

string += "\""
string += "\n\t}"
string += "\n]"

def export_string_to_txt_file(string, file_name):
    with open(file_name, "w") as file:
        file.write(string)

file_name = "complete.pilot.tl.json"        
export_string_to_txt_file(string, file_name)

    
dataFrameChoice = pd.DataFrame.from_dict(choiceSet)
dataFrameChoice.to_csv('choice_pilot_full.csv')
dataFrameChoice.to_pickle("choiceset_pilot_full.pkl")

dataframeD = pd.DataFrame.from_dict(data)
dataframeD['avg'] = dataframeD.mean(axis=1)
dataframeD['var'] = dataframeD.var(axis=1)
dataframeD.to_csv('asset.csv')

dataframeOcc = pd.DataFrame.from_dict(occurence)
dataframeOcc.to_csv('occurence.csv')