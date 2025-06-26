# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:52:05 2024

@author: benja
"""


data_copier = {}
for stage in range(10):
    stage_key = f"stage_{stage}"
    data_copier[stage_key] = {
        "CRRA_-1.5": 1,
        "CRRA_0": 2,
        "CRRA_1": 3,
        "CRRA_3": 4,
        "CRRA_6": 5
    }




string = "const data = {"
string += "\n"

for i in range(1, len(data_copier)+1) :
    stage = "stage_" + str(i)
    stagestring = "stage_" + str(i-1)
    string += "\t"
    string += f'{stagestring}: '
    string += '{'
    string += "\n"
    for key in data_copier[stagestring].keys():
        string += "\t\t"
        string += f'"{key}": '
        string += str(f'{data_copier[stagestring][key]},')
        string += "\n"
    string += "\t\t"
    string += "},"
    string += "\n"
string += "};"
               
print(string)

def export_string_to_txt_file(string, file_name):
    with open(file_name, "w") as file:
        file.write(string)

def replace_nan(input_string):
    return input_string.replace('nan', 'NaN')

string = replace_nan(string)
file_name = f"javastrings_bots_series_{series}_{itera}.txt"
subfolder = 'Botdata'
file_path = os.path.join(subfolder, file_name)
file_path = os.path.join(folder_path, file_path)
export_string_to_txt_file(string, file_path)