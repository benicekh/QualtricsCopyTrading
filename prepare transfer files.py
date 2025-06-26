# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:33:37 2025

@author: benja
"""

import os
import re

# Set the root folder
folder_path = r"D:\Surfdrive\Projects\Copy Trading\Trading game"

# Define subfolder paths
botdata_folder = os.path.join(folder_path, "Botdata")
paths_folder = os.path.join(folder_path, "Paths")
tldata_folder = os.path.join(folder_path, "TLData")
combined_folder = os.path.join(folder_path, "combined")

# Ensure the combined folder exists
os.makedirs(combined_folder, exist_ok=True)

# Process files by matching their numbers
for file in os.listdir(botdata_folder):
    match = re.match(r"javastrings_bots_series_(high|low)_(\d+)\.txt", file)
    if not match:
        continue  # Skip non-matching files

    category, number = match.groups()  # Extract "high/low" and the number
    bot_file = os.path.join(botdata_folder, file)
    price_file = os.path.join(paths_folder, f"javastrings_price_series_{number}.txt")
    tl_file = os.path.join(tldata_folder, f"javastrings_TLs_series_{number}.txt")

    # Read bot series
    with open(bot_file, "r", encoding="utf-8") as f:
        bot_content = f.read()

    # Read price series and remove the first list
    with open(price_file, "r", encoding="utf-8") as f:
        price_content = f.readlines()

    new_price_content = []
    found_first_list = False
    for line in price_content:
        if "pricePaths.path" in line and not found_first_list:
            found_first_list = True  # Skip the first price list
        else:
            new_price_content.append(line)

    modified_price_content = "".join(new_price_content)

    # Read TLs series
    with open(tl_file, "r", encoding="utf-8") as f:
        tl_content = f.read()

    # Qualtrics snippet for the full version (WITH TLs)
    qualtrics_snippet_with_tl = """
    Qualtrics.SurveyEngine.setEmbeddedData("TLs", JSON.stringify(TLs));
    Qualtrics.SurveyEngine.setEmbeddedData("botdata", JSON.stringify(data));
    Qualtrics.SurveyEngine.setEmbeddedData("pricepaths", JSON.stringify(pricePaths));
    """

    # Qualtrics snippet for the noTL version (WITHOUT TLs)
    qualtrics_snippet_without_tl = """
    Qualtrics.SurveyEngine.setEmbeddedData("botdata", JSON.stringify(data));
    Qualtrics.SurveyEngine.setEmbeddedData("pricepaths", JSON.stringify(pricePaths));
    """

    # Combine contents (WITH TLs)
    combined_content_with_tl = bot_content + "\n" + modified_price_content + "\n" + tl_content + "\n" + qualtrics_snippet_with_tl

    # Combine contents (WITHOUT TLs)
    combined_content_without_tl = bot_content + "\n" + modified_price_content + "\n" + qualtrics_snippet_without_tl

    # Save the files with correct naming
    output_file_with_tl = os.path.join(combined_folder, f"javastrings_bots_series_{category}_{number}.txt")
    output_file_without_tl = os.path.join(combined_folder, f"javastrings_bots_series_{category}_{number}_noTL.txt")

    with open(output_file_with_tl, "w", encoding="utf-8") as f:
        f.write(combined_content_with_tl)

    with open(output_file_without_tl, "w", encoding="utf-8") as f:
        f.write(combined_content_without_tl)

    print(f"Saved: {output_file_with_tl}")
    print(f"Saved: {output_file_without_tl}")