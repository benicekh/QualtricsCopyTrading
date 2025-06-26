# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:43:01 2025

@author: benja
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import log
import os
import pandas as pd
import re
import pickle
#from mpmath import *
#mp.dps=300

#Set the seed to make random generation reproducible
seed = 3062024
np.random.seed(seed)
combinations = 100000 #amount of pathseries to created
paths = 10000 #amount of different paths to be created

### data created using seed=3062024,combinations=100000, paths = 10000

folder_paths = [
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game"
]

# Automatically select the first existing folder
folder_path = next((path for path in folder_paths if os.path.exists(path)), None)

if folder_path is None:
    raise FileNotFoundError("None of the specified folder paths exist.")

print(f"Using folder path: {folder_path}")
price_paths_path = os.path.join(folder_path, "Paths")

#Setup

results = {}
result_dict = {}
result_dict_returns = {}
final_results = {}
final_results_returns = {}
collection = {}
collection["Rank"] = {}
collection["Gain"] = {}
for key in collection.keys():
    collection[key]["CRRA_-1.5"] = pd.DataFrame()
    collection[key]["CRRA_0"] = pd.DataFrame()
    collection[key]["CRRA_1"] = pd.DataFrame()
    collection[key]["CRRA_3"] = pd.DataFrame()
    collection[key]["CRRA_6"] = pd.DataFrame()
    collection[key]["CRRA_7"] = pd.DataFrame()
# Functions
## Saving strings to txt
def export_string_to_txt_file(string, file_name):
    with open(file_name, "w") as file:
        file.write(stringp)

##Creating new Z and DeltaS arrays that are needed for profit and belief calculation, based on trimmed price array S
def compute_Z_new(S):
    # Compute the differences between consecutive rows
    differences = np.diff(S, axis=0)
    # Create Zn based on the sign of the differences
    Zn = np.where(differences > 0, 1, 0)
    
    
    return Zn


def compute_DeltaS_new(S):
    differences = np.diff(S, axis=0)    
    return differences

 
## Bayesian updating
#Function that takes the prior belief about the prob of being in the good state and updating it depending on the direction of the price change
def p_hat(p,z, omega):
    if skill == 0:
        num = (((0.5+ch)**z*(0.5-ch)**(1-z))**omega)*p
        denom = num + (((0.5-ch)**z*(0.5+ch)**(1-z))**omega)*(1-p)
    else:
        num = (((0.5+ch)**z*(0.5-ch)**(1-z)))*p
        denom = num + (((0.5-ch)**z*(0.5+ch)**(1-z)))*(1-p)
    return num/denom

def p_update(p, z, omega, gamma):
    prob = p_hat(p, z, omega)
    if skill == 0:
        return (1-q)*prob + q*(1-prob)
    else:
        change = q*gamma
        return (1-change)*prob + change*(1-prob)

#Functions to calculate expected utility
# Adjust the utility function here
def u(x, element):
    if x != 0:
        x = x/1000
        if element < 1 and element != 1:
            num = ((x**(1 - element))-1)
            den = (1 - element)
        elif element == 1:
            num = log(x)
            den = 1 
        else:
            num = ((x**(1 - element))-1)
            den = (1 - element)              
        return num / den
    else:
        if element != 1:
            return 0
        else:
            return float('-inf')  # Return negative infinity for log(0) case 

def calcEU(c, a, p, eta): 
    #upG =  p*(0.5+ch)*(1/3)
    #upB = (1-p)*(0.5-ch)*(1/3)
    #dG = p*(0.5-ch)*(1/3)
    #dB = (1-p)*(0.5+ch)*(1/3)
    #return sum([upG * u(c + a * k, eta) + upB * u(c + a * k, eta) + dG * u(c + a * (-1) * k, eta) + dB * u(c + a * (-1) * k, eta) for k in ks]) if a != 0 else u(c, eta)
    #return (1+2*p)/(12)*sum([u(c + a * k, eta) for k in ks]) + (3-2*p)/(12)*sum([u(c + a * (-1) * k, eta) for k in ks]) if a != 0 else u(c, eta)
    #return ((8*p*ch)+2-(4*ch))/(12)*sum([u(c + a * k, eta) for k in ks]) + (2+(4*ch)-(8*p*ch))/(12)*sum([u(c + a * (-1) * k, eta) for k in ks]) if a != 0 else u(c, eta)
    pos = []
    neg = []
    for k in ks:
        tempPos = c + a * k
        pos.append(u(tempPos, eta))
        temp = c + a * (-1) * k
        tempNeg = u(max(temp, 0), eta)
        neg.append(tempNeg)        
    sum1 = sum([p for p in pos])
    sum2 = sum([n for n in neg])
    # print(neg)
    # print(pos)
    # print(sum1)
    # print(sum2)
    up = ((8*p*ch)+2-(4*ch))/(12)
    down = (2+(4*ch)-(8*p*ch))/(12)
    return up*sum1 + down*sum2 if a != 0 else u(c, eta)

# Function that compares utility functions with concrete values for assets
# Calculates how many assets DM would buy
def assetsBought(c, p, eta, a_max):
    EU_values = [calcEU(c, a, p, eta) for a in range(a_max+1)]
    max_value = max(EU_values)
    max_index = [i for i, j in enumerate(EU_values) if j == max_value]
    asset = max(max_index) if max_index else 0
    return asset

def round_var(var):
    if isinstance(var, (np.float64, float)):
        var = np.array([var])
    elif isinstance(var, list):
        var = np.array(var)
    elif isinstance(var, (pd.Series, pd.DataFrame)):
        var = var
    else:
        var = pd.Series(var)
    
    # Ensure the Series/DataFrame is converted to a numeric type
    if isinstance(var, pd.Series):
        var = pd.to_numeric(var, errors='coerce')
    elif isinstance(var, pd.DataFrame):
        var = var.apply(pd.to_numeric, errors='coerce')
    
    # Apply rounding
    var = var.round(2) if isinstance(var, (pd.Series, pd.DataFrame)) else np.round(var, 2)

    return var

def create_bots(num_players,num_series, column_names):
    data = {}
    for i in range(1, num_series + 1):
        series_name = f'{i}'
        data[series_name] = {}
        for j in range(1, num_players + 1):
            player_name = f'player_{j}'
            data[series_name][player_name] = pd.DataFrame(columns=column_names)
    return data
    
def populate_dataframes(data_player, n, m, array):
    for key, df in data_player.items():
        # Create the 'r' column
        r_values = []
        for i in range(m):
            r_values.extend(list(range(n)))
        df['r'] = r_values
        
        # Create the 'stg' column
        stg_values = []
        for i in range(len(r_values)):
            stg_values.append(i // n)
        df['stg'] = stg_values
        
        # Add the 'phase' column
        df['phase'] = 'regular'
        
        # preset Rank for later
        df['Rank'] = 1
        
        # Combine the first m columns of the first n rows of the array and insert into column 'p'
        combined_cols = array[:n, :m].reshape(-1, order="F")
        df['p'] = combined_cols
        
    return data_player

# Collection of which variables/dictionaries/etc. ... contain what because I'm forgetful
# and not commenting as much as I should
# also not good at naming stuff
Legend = {}
Legend["S"] = "Contains the actual price paths"
Legend["q"] = "Probability of state switch"
Legend["ch"] = "0.5 - ch, probability that price chnages opposite to what state dictates"

## Setting up the simulation
#for q in [0.1, 0.2, 0.3, 0.4]:
for q in [0.15]:
    #for ch in [0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
    for ch in [0.15]:
        T = 40 #number of periods
        M = paths #number of iterations to run
        q = q #probability to switch states
        ch = ch #governs the chance to get the opposite price change per state, e.g. down in good state, Pr = 0.5-ch
        S0 = 250 #initial stock price
        c = 2500 #cash endowment
        cash = c #for calculation of gain array
        ks = [5, 10, 15] # Define the possible outcomes
        probabilities = [1/3, 1/3, 1/3] # Set the probabilities for each outcome (uniform distribution)
        
        ##########################################################################
        outputtxt = 1 #switch for whether text outputs should be created or not
        
        ##Interval of possible risk attitude coefficients eta
        #etas = np.round(np.arange(-5, 5.1, 2.5), decimals=2) #interval of risk attitude coefficients we want to look at
        #etas = np.insert(etas, 4, 4)
        etas = [-1.5,0,1,3,6]
        
        # Skill for Bayesian updating
        # Will implement skill differences via exponent that amplifies or reduces the informational content of the price change 
        # omega = 1 is the correct updater case (Bayesian type), omega < 1 underestimates (sceptical types), omega > 1 overestimates (gullible types)
        # from Palfrey, T. R., & Wang, S. W. (2012). Speculative overpricing in asset markets with information flows. Econometrica, 80(5), 1937-1976.
        #omega = np.round(np.arange(0, 2.21, 0.4), decimals=2)
        omega=[1.0]
        
        # Alternative Skill implementation
        # Misconception about the probability of states to change
        # More skilled investors might be able to judge chnages in the market better
        # some might underestimate how likely it is to switch from bear to bull market and vice versa
        # some might overestimate 
        # calculated by taking the product of real change probability and skill factor
        # above 1 will overestimate change, should not be above 2 depending on ch
        gamma=[1.0]
        
        # Toggle which skill model to use
        # omega: skill = 0
        # gamma: skill = 1
        skill = 0
        
        #First generating the price paths
        # Generating the state process
        theta = np.zeros((T+1,M))
        theta0 = stats.bernoulli.rvs(0.5, size=M) #initial state for each of the $M$ iterati
        Deltatheta = stats.bernoulli.rvs(q, size=(T,M)) #generate all the Δθ's in one go
        theta[0] = theta0 #initialised state for every iteration, vectorised
        #and now generating the θ_1,...,θ_T via θ_(t+1) = θ_t + Δθ_t in vectorised form
        for t in range(T):
            theta[t+1] = theta[t] + Deltatheta[t]
        
        DeltaS = np.random.choice(ks, size=(T,M), p=probabilities) # generating the possible price changes
        P = 0.5 + ch*(-1)**theta #generate all the probabilities for the direction of the price changes
        Z = stats.bernoulli.rvs(P) #generate the directional variables
        S = S0*np.ones((T+1,M)) #initialise
        for t in range(T):
            S[t+1] = np.maximum(0, S[t] - (-1)**Z[t] * DeltaS[t])


T, M = S.shape #adjusting dimension variables
T = T-1   

beliefs = {}
for o in omega:   
    variable_name = str(o)
    g = 1
    belief = 0.5*np.ones((T+1,M))
    for t in range(T):
        belief[t+1] = p_update(belief[t], Z[t], o, g)
    beliefs[variable_name] = belief




Z = compute_Z_new(S)
DeltaS = np.abs(compute_DeltaS_new(S))
# Generate max possible amount of assets
a_max = np.ones((T,M))
# 0 if price is 0
a_max = np.where(S == 0, 0, np.divide(c, S))
c_array = np.ones((T+1,M))*cash
gain_array = np.zeros((T+1,M))

key_list = []
longkey_list = []
value_list = []
sum_list = []
sd_list = []
ind_list = []
plotdata = {}
profits = {}
gains = {}
assets = {}
cashs = {}
assetmaxes = {}
for e in etas:
    etastr = str(e)
    gains[etastr] = {}
    profits[etastr] = {}
    assets[etastr] = {}
    cashs[etastr] = {}
    assetmaxes[etastr] = {}
for o in omega:
    string = str(o)
    belief = beliefs[string]
    beliefstr = str(o)
    plot_key = []
    plot_value = [] 
    for e in etas:
        eta = e
        etastr = str(e)
        #print(eta)
        boughtAssets = 0*np.ones((T+1,M))
        profit = 0*np.ones((T,M))
        for m in range(a_max.shape[1]):
            for t in range(a_max.shape[0]-1): #calculating the choice of assets bought for every cell
                pbelief = belief[t, m]   #this is what makes it slow, need to figure out how to vectorise
                a_maxes = int(np.floor(a_max[t ,m])) 
                c = c_array[t, m]
                bought = assetsBought(c, pbelief, eta, a_maxes)
                boughtAssets[t, m] = bought
                profit[t,m] = boughtAssets[t,m]*(DeltaS[t,m]*(-1)*(-1)**Z[t,m])
                prof = profit[t,m]
                c_array[t+1, m] = c + prof
                gain_array[t+1,m] = c_array[t+1,m] - cash
                if t < (a_max.shape[0]-1):
                    a_max[t+1,m] = np.where(S[t+1,m] == 0, 0, np.divide(c_array[t+1,m], S[t+1,m]))
        gains[etastr][beliefstr] = gain_array.copy()
        assets[etastr][beliefstr] =  boughtAssets.copy()
        assetmaxes[etastr][beliefstr] = a_max.copy()
        cashs[etastr][beliefstr] =  c_array.copy()
        
        #profit = 0*np.ones((T,M))
        #for t in range(T):        
        #    profit[t] = boughtAssets[t]*(DeltaS[t]*(-1)*(-1)**Z[t])

        column_sums = np.sum(profit, axis=0) #calculating and storing values for the graphs
        column_avg = np.mean(column_sums, axis=0)
        column_sd = np.std(column_sums, axis=0)
        individual_gains = profit.flatten()
        key = str(etastr)
        key_list.append(key)
        longkey = f"Switch:{q}-Adverse:{np.round(0.5-ch, decimals=2)}-Omega:{o}-Eta:{e}"
        longkey_list.append(longkey)
        plot_key.append(key)
        #plot_value.append(individual_gains)
        plot_value.append(column_sums)
        value_list.append(column_sums)
        sum_list.append(column_avg)
        ind_list.append(individual_gains)
        sd_list.append(column_sd)
        profits[etastr][beliefstr] = profit
        print(f"loop for omega: {o} and eta: {e}")
    plot_dict = dict(zip(plot_key, plot_value)) # Graphing commands
    plotdata[beliefstr] = plot_value
    plot_dataframe = pd.DataFrame.from_dict(plot_dict, orient='index').transpose()

    
# Get sorted keys for consistent order
sorted_keys = sorted(profits.keys())

# Get number of columns
n_cols = next(iter(profits.values()))['1.0'].shape[1]

# Initialize storage for summed values
performance = {key: [] for key in sorted_keys}

# Calculate sums per column
for col in range(n_cols):
    for key in sorted_keys:
        arr = profits[key]['1.0']
        col_sum = arr[:, col].sum()
        performance[key].append(col_sum)

# Convert to performance array
performance_array = np.array([performance[key] for key in sorted_keys])

# Now, rank across keys per column
rank_array = np.zeros_like(performance_array, dtype=int)

for col in range(n_cols):
    # Get sums for this column across keys
    col_sums = {key: performance[key][col] for key in sorted_keys}
    # Sort keys by sum (highest first)
    sorted_by_sum = sorted(col_sums, key=col_sums.get, reverse=True)
    # Assign ranks
    for rank, key in enumerate(sorted_by_sum):
        rank_array[sorted_keys.index(key), col] = rank + 1

Legend["sorted_keys"] = "label for arrays like performance_array, shows what row corresponds to what bot"
Legend["performance_array"] = "final gain per price path per bot, bot order in sorted_key"
Legend["rank_array"] = "rank per price path per bot, bot order in sorted_key"

print("Row labels (keys):")
print(np.array(sorted_keys))

print("\nPerformance array (column sums):")
print(performance_array)

print("\nRank array (ranks of sums):")
print(rank_array)

high_risk_best = []
low_risk_best = []

# Loop over all columns
for col in range(rank_array.shape[1]):
    if (rank_array[0, col] == 1 and
        rank_array[2, col] == 3 and
        rank_array[4, col] == 5):
        high_risk_best.append(col)
    
    if (rank_array[0, col] == 5 and
        rank_array[2, col] == 3 and
        rank_array[4, col] == 1):
        low_risk_best.append(col)

Legend["high_risk_best"] = "all price paths in which high risk bot is best, lowest risk bot is worst and CRRA 1 is 3rd"
Legend["low_risk_best"] = "all price paths in which lowest risk bot is best, highest risk bot is worst and CRRA 1 is 3rd"


print("high_risk_best column indices:")
print(high_risk_best)

print("\nlow_risk_best column indices:")
print(low_risk_best)

# Function to get top decile column indices
def get_top_decile(columns, diff_values, total_cols):
    num_top = max(1, int(np.ceil(len(columns) * 0.1)))  # at least 1
    sorted_cols = [col for col, _ in sorted(diff_values.items(), key=lambda x: x[1], reverse=True)]
    return sorted_cols[:num_top]

# --- High risk ---
high_diffs = {}
for col in high_risk_best:
    diff = performance_array[0, col] - performance_array[-1, col]
    high_diffs[col] = diff

top10_high_risk_best = get_top_decile(high_risk_best, high_diffs, performance_array.shape[1])

# --- Low risk ---
low_diffs = {}
for col in low_risk_best:
    diff = performance_array[-1, col] - performance_array[0, col]
    low_diffs[col] = diff

top10_low_risk_best = get_top_decile(low_risk_best, low_diffs, performance_array.shape[1])

Legend["top10_high_risk_best"] = "from list of high_risk_best, top 10 in terms of performance difference between 1st and last"
Legend["top10_low_risk_best"] = "from list of low_risk_best, top 10 in terms of performance difference between 1st and last"


print("Top 10% high_risk_best column indices:")
print(top10_high_risk_best)

print("\nTop 10% low_risk_best column indices:")
print(top10_low_risk_best)

risk_start = np.random.choice(top10_high_risk_best, 1, replace=False)
no_risk_start = np.random.choice(top10_low_risk_best, 1, replace=False)

#Construct different pathseries from generated paths
# Initialize empty array: (10 selections, 10 iterations)
final_selection_matrix = np.empty((10, combinations), dtype=int)
final_zeroes_matrix = np.empty((5, combinations), dtype=int)
final_ones_matrix = np.empty((5, combinations), dtype=int)

for i in range(combinations):
    # Find indices of 0s and 1s
    zero_indices = np.where(theta0 == 0)[0]
    one_indices = np.where(theta0 == 1)[0]
    
    # Randomly select 5 indices each
    selected_zero_indices = np.random.choice(zero_indices, 5, replace=False)
    selected_one_indices = np.random.choice(one_indices, 5, replace=False)
    
    # Combine and shuffle
    selected_indices = np.concatenate([selected_zero_indices, selected_one_indices])
    np.random.shuffle(selected_indices)
    
    # Store as column in the matrix
    final_selection_matrix[:, i] = selected_indices
    final_zeroes_matrix[:, i] = selected_zero_indices
    final_ones_matrix[:, i] = selected_one_indices
    
performance_all_paths = {}
overall_performance_all_paths = {}
overall_rank_all_paths = {}
rank_all_paths = {}

for key_idx, key in enumerate(sorted_keys):
    # Prepare empty arrays of same shape as final_selection_matrix
    perf_matrix = np.empty_like(final_selection_matrix, dtype=performance_array.dtype)
    rank_matrix = np.empty_like(final_selection_matrix, dtype=rank_array.dtype)
    
    # Fill in the values
    for i in range(final_selection_matrix.shape[0]):  # rows (10)
        for j in range(final_selection_matrix.shape[1]):  # cols (100000)
            col_idx = final_selection_matrix[i, j]
            perf_matrix[i, j] = performance_array[key_idx, col_idx]
            rank_matrix[i, j] = rank_array[key_idx, col_idx]
    
    # Store original performance and rank matrices
    performance_all_paths[key] = perf_matrix
    rank_all_paths[key] = rank_matrix

    # Compute cumulative performance matrix (row-wise sum)
    overall_performance_all_paths[key] = np.cumsum(perf_matrix, axis=0)

# Now compute ranks across keys at each (i, j) position
keys = sorted(performance_all_paths.keys())
shape = final_selection_matrix.shape
overall_rank_all_paths = {key: np.empty(shape, dtype=int) for key in keys}

for i in range(shape[0]):
    for j in range(shape[1]):
        # Collect values from all keys at position (i, j)
        values = np.array([overall_performance_all_paths[key][i, j] for key in keys])
        
        # Rank them: rank 1 = highest value (best), so we use argsort in descending order
        ranks = (-values).argsort().argsort() + 1  # Convert to 1-based rank

        for k_idx, key in enumerate(keys):
            overall_rank_all_paths[key][i, j] = ranks[k_idx]

# Create labels for columns
col_labels = [f'{i}' for i in range(final_selection_matrix.shape[1])]

# Initialize storage for DataFrames
perf_sums_data = {}
rank_vars_data = {}

for key in sorted_keys:
    # Get matrices
    perf_matrix = performance_all_paths[key]
    rank_matrix = rank_all_paths[key]
    
    # Compute column sums and variances
    col_sums = np.sum(perf_matrix, axis=0)
    col_vars = np.var(rank_matrix, axis=0)
    
    # Store in dictionary
    perf_sums_data[key] = col_sums
    rank_vars_data[key] = col_vars

# Convert to DataFrames
perf_sums_df = pd.DataFrame.from_dict(perf_sums_data, orient='index', columns=col_labels)
rank_vars_df = pd.DataFrame.from_dict(rank_vars_data, orient='index', columns=col_labels)


col_averages = rank_vars_df.mean(axis=0)

# Get sorted column indices (ascending order)
sorted_cols = col_averages.sort_values().index.tolist()
total_cols = len(sorted_cols)
top_n = max(1, int(total_cols * 0.2))  # at least 1

# Convert column labels (e.g., 'iter_1') to numeric positions
col_positions = {label: idx for idx, label in enumerate(rank_vars_df.columns)}

# Get bottom, middle, top 20%
bottom_cols = sorted_cols[:top_n]
top_cols = sorted_cols[-top_n:]
middle_start = (total_cols - top_n) // 2
middle_cols = sorted_cols[middle_start:middle_start + top_n]

# Map to column numbers
bottom20_rank_var = [col_positions[col] for col in bottom_cols]
middle20_rank_var = [col_positions[col] for col in middle_cols]
top20_rank_var = [col_positions[col] for col in top_cols]

print("Bottom 20% column numbers (lowest averages):")
print(bottom20_rank_var)

print("\nMiddle 20% column numbers (middle averages):")
print(middle20_rank_var)

print("\nTop 20% column numbers (highest averages):")
print(top20_rank_var)

# Create labels for columns (as strings)
col_labels = [f'{i}' for i in range(final_selection_matrix.shape[1])]

# Initialize storage for overall rank variances
overall_rank_vars_data = {}

for key in sorted_keys:
    overall_rank_matrix = overall_rank_all_paths[key]
    
    # Compute column variances
    col_vars = np.var(overall_rank_matrix, axis=0)
    
    # Store in dictionary
    overall_rank_vars_data[key] = col_vars

# Convert to DataFrame
overall_rank_vars_df = pd.DataFrame.from_dict(overall_rank_vars_data, orient='index', columns=col_labels)

# Compute average variance per column
overall_col_averages = overall_rank_vars_df.mean(axis=0)

# Sort column labels by average variance
overall_sorted_cols = overall_col_averages.sort_values().index.tolist()
total_cols = len(overall_sorted_cols)
top_n = max(1, int(total_cols * 0.1))  # 10% of columns

# Convert column labels (strings) to numeric positions
overall_col_positions = {label: idx for idx, label in enumerate(overall_rank_vars_df.columns)}

# Get bottom and top 10% columns
bottom_cols = overall_sorted_cols[:top_n]
top_cols = overall_sorted_cols[-top_n:]

# Map to column indices (int)
bottom10pct_overall_rank_var = [overall_col_positions[col] for col in bottom_cols]
top10pct_overall_rank_var = [overall_col_positions[col] for col in top_cols]

print("Bottom 10% column indices (most stable overall ranks):")
print(bottom10pct_overall_rank_var)

print("\nTop 10% column indices (most volatile overall ranks):")
print(top10pct_overall_rank_var)


# Ranking overall performance
perf_ranks_df = perf_sums_df.rank(axis=0, method='min', ascending=False)
Legend["perf_ranks_df"] = "Ranking of end of path series performance, uses ordering from sorted_keys"

# Overall performance ranking
high_risk_best_overall = []
low_risk_best_overall = []

for col_idx, col in enumerate(perf_ranks_df.columns):
    first_rank = perf_ranks_df.iloc[0, col_idx]
    last_rank = perf_ranks_df.iloc[-1, col_idx]
    
    if first_rank == 1 and last_rank == perf_ranks_df.shape[0]:
        high_risk_best_overall.append(col_idx)
    if first_rank == perf_ranks_df.shape[0] and last_rank == 1:
        low_risk_best_overall.append(col_idx)

# Convert all lists to sets for fast intersection
bottom20_set = set(bottom20_rank_var)
middle20_set = set(middle20_rank_var)
top20_set = set(top20_rank_var)
high_risk_set = set(high_risk_best_overall)
low_risk_set = set(low_risk_best_overall)
bottom10_set = set(bottom10pct_overall_rank_var)
top10_set = set(top10pct_overall_rank_var)

# Build the dictionary
#overlap_dict = {
#    'bottom_high': sorted(bottom20_set & high_risk_set),
#    'bottom_low': sorted(bottom20_set & low_risk_set),
#    'middle_high': sorted(middle20_set & high_risk_set),
#    'middle_low': sorted(middle20_set & low_risk_set),
#    'top_high': sorted(top20_set & high_risk_set),
#    'top_low': sorted(top20_set & low_risk_set)
#}
#
overlap_dict = {
    'bottom_high': sorted(bottom10_set & high_risk_set),
    'bottom_low': sorted(bottom10_set & low_risk_set),
    'top_high': sorted(top10_set & high_risk_set),
    'top_low': sorted(top10_set & low_risk_set)
}

# Print results
for key, indices in overlap_dict.items():
    print(f"{key}: {indices}")
    
overlap_dict_choice = {}

for key, indices in overlap_dict.items():
    indices_array = np.array(indices)
    if len(indices) >= 2:
        # Randomly select 2 unique values
        chosen = np.random.choice(indices_array, 2, replace=False).tolist()
    else:
        # Take all if fewer than 2
        chosen = indices
    overlap_dict_choice[key] = chosen

# Print the results
for key, chosen_values in overlap_dict_choice.items():
    print(f"{key}: {chosen_values}")
    
risk_parts = []
norisk_parts = []
risk_keys = []
norisk_keys = []

for key, col_indices in overlap_dict_choice.items():
    if len(col_indices) < 2:
        continue  # skip incomplete pairs

    col1, col2 = col_indices

    # Risk path: first column + risk_start
    col1_values = final_selection_matrix[:, col1]
    col1_with_risk = np.insert(col1_values, 0, risk_start)
    risk_parts.append(col1_with_risk)
    risk_keys.append(f"risk_start-{key}")

    # No-risk path: second column + no_risk_start
    col2_values = final_selection_matrix[:, col2]
    col2_with_norisk = np.insert(col2_values, 0, no_risk_start)
    norisk_parts.append(col2_with_norisk)
    norisk_keys.append(f"no_risk_start-{key}")

# Concatenate all parts: first all risk, then all no-risk
all_parts = risk_parts + norisk_parts
final_selection = np.concatenate(all_parts).reshape(-1, 1)

# Create the final_selection_keys list
final_selection_keys = risk_keys + norisk_keys

print("Final selection shape:", final_selection.shape)
print(final_selection)

print("\nFinal selection keys order:")
print(final_selection_keys)

selected_indices = final_selection.flatten()
selected_S = S[:, selected_indices]

def export_string_to_txt_file(string, file_name):
    with open(file_name, "w") as file:
        file.write(stringp)

subfolder = 'Paths'
batch_size = 11  # Number of columns per file
num_cols = selected_S.shape[1]
i = 0
# Loop over column batches
for start_col in range(0, num_cols, batch_size):
    key_label = final_selection_keys[i].replace(' ', '_')  # Clean up spaces if needed
    i += 1
    stringp = ""
    end_col = min(start_col + batch_size, num_cols)
    
    string = "const pricePaths = {}"
    string += "\n"
    
    for col in range(start_col, end_col):
        pathname = f"pricePaths.path{col+1}"
        stringp += f"{pathname} = ["
        
        for row in range(selected_S.shape[0]):
            stringp += f"{selected_S[row, col]},"
        
        stringp += "]\n"
    count = i
    
    file_name = f"javastrings_price_series_{key_label}.txt"
    file_path = os.path.join(subfolder, file_name)
    file_path = os.path.join(folder_path, file_path)
    export_string_to_txt_file(stringp, file_path)


print("Export complete!")

subfolder = 'Paths'
file_path = os.path.join(folder_path, subfolder)
file_path = os.path.join(file_path, f"all_paths.csv")
np.savetxt(file_path, S, delimiter=',', fmt='%d')


#S = pick_random_paths(S,101)
file_path = os.path.join(folder_path, subfolder)
file_path = os.path.join(file_path, f"picked_paths.csv")
np.savetxt(file_path, S, delimiter=',', fmt='%d')

T, M = S.shape #adjusting dimension variables
T = T-1   

datadump = {}
datadump["all_generated_paths"] = S
S = selected_S #Overwriting original price array
datadump["risk_start"] = risk_start
datadump["no_risk_start"] = no_risk_start
datadump["selected_paths"] = selected_S
datadump["selected_indeces"] = final_selection
datadump["type"] = final_selection_keys


###### Done picking price paths
### Graph Path rank changes
final_S_rank = {}
selected_indices_flat = np.array(selected_indices).flatten()
for key_idx, key in enumerate(sorted_keys):
    # Collect rank values for this key
    rank_values = [rank_array[key_idx, col_idx] for col_idx in selected_indices_flat]
    final_S_rank[key] = rank_values
final_S_rank_split = {}

block_size = 11
num_blocks = len(final_selection_keys)

for key in sorted_keys:
    # Get the full list of rank values for this key
    rank_values = final_S_rank[key]
    
    # Initialize the nested dictionary
    final_S_rank_split[key] = {}
    
    for block_idx, block_key in enumerate(final_selection_keys):
        start = block_idx * block_size
        end = start + block_size
        
        # Slice the 11 values for this block
        chunk = rank_values[start:end]
        
        # Store under the nested key
        final_S_rank_split[key][block_key] = chunk

# Example output check
for outer_key, nested_dict in final_S_rank_split.items():
    print(f"\nMain Key: {outer_key}")
    for inner_key, chunk in nested_dict.items():
        print(f"  {inner_key}: {chunk}")
##### Plot the performances
# Get block keys
block_keys = list(next(iter(final_S_rank_split.values())).keys())

for block_key in block_keys:
    plt.figure(figsize=(10, 10))  # taller figure for space
    
    # Step 1: Parse block_key
    prefix, core_key = block_key.split('-', 1)
    col_list = overlap_dict_choice.get(core_key, None)
    
    if col_list is None:
        print(f"Warning: No entry in overlap_dict_choice for {core_key}. Skipping final point.")
        matched_col = None
    else:
        matched_col = col_list[0] if prefix == 'risk_start' else col_list[1]
    
    # Prepare raw data matrix
    table_data = []
    col_labels = [str(i) for i in range(1, 12)] + ['Final']
    
    for main_key in sorted_keys:
        y_values = final_S_rank_split[main_key][block_key].copy()
        x_values = list(range(1, len(y_values) + 1))
        
        # Plot the main line
        plt.plot(x_values, y_values, label=main_key)
        
        # Add final point
        if matched_col is not None:
            final_value = int(perf_ranks_df.loc[main_key].iloc[matched_col])
            plt.scatter(12, final_value, marker='o')
            y_values.append(final_value)
        else:
            y_values.append('N/A')
        
        table_data.append(y_values)
    
    # X-axis labels
    plt.xticks(list(range(1, 12)) + [12], labels=[str(i) for i in range(1, 12)] + ['Final'])
    
    # Y-axis ticks
    plt.yticks([1, 2, 3, 4, 5])
    
    plt.title(f"Rank Over Time with Final Point - {block_key}")
    plt.xlabel("Time (Position)")
    plt.ylabel("Rank Value")
    
    # Legend between plot and table
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # Add table at the very bottom
    table = plt.table(cellText=table_data,
                      rowLabels=sorted_keys,
                      colLabels=col_labels,
                      loc='bottom',
                      bbox=[0.0, -0.6, 1, 0.3])  # push table down
    
    # Adjust plot to make space
    plt.subplots_adjust(left=0.2, bottom=0.4)  # increase bottom space
    
    # Save the plot
    save_path = os.path.join(price_paths_path, f"{block_key}.png")
    plt.savefig(save_path, bbox_inches='tight')

    
    print(f"Saved plot with legend and table: {save_path}")

###### Create and output botdata

T, M = S.shape #adjusting dimension variables
T = T-1   
Z = compute_Z_new(S)
DeltaS = np.abs(compute_DeltaS_new(S))

beliefs = {}

for o in omega:

    variable_name = str(o)
    g = 1
    belief = 0.5*np.ones((T+1,M))
    for t in range(T):
        belief[t+1] = p_update(belief[t], Z[t], o, g)
    beliefs[variable_name] = belief



# Generate max possible amount of assets
a_max = np.ones((T,M))
# 0 if price is 0
a_max = np.where(S == 0, 0, np.divide(c, S))
c_array = np.ones((T+1,M))*cash
gain_array = np.zeros((T+1,M))

# This is slow!
# Calculates price path profits for different paths, looped over omega/eta combinations
# Plots Boxplots per omega for every eta

key_list = []
longkey_list = []
value_list = []
sum_list = []
sd_list = []
ind_list = []
plotdata = {}
profits = {}
gains = {}
assets = {}
cashs = {}
assetmaxes = {}
for e in etas:
    etastr = str(e)
    gains[etastr] = {}
    profits[etastr] = {}
    assets[etastr] = {}
    cashs[etastr] = {}
    assetmaxes[etastr] = {}

for o in omega:
    string = str(o)
    belief = beliefs[string]
    beliefstr = str(o)
    plot_key = []
    plot_value = [] 
    for e in etas:
        eta = e
        etastr = str(e)
        #print(eta)
        boughtAssets = 0*np.ones((T+1,M))
        profit = 0*np.ones((T,M))
        for m in range(a_max.shape[1]):
            for t in range(a_max.shape[0]-1): #calculating the choice of assets bought for every cell
                pbelief = belief[t, m]   #this is what makes it slow, need to figure out how to vectorise
                a_maxes = int(np.floor(a_max[t ,m])) 
                c = c_array[t, m]
                bought = assetsBought(c, pbelief, eta, a_maxes)
                boughtAssets[t, m] = bought
                profit[t,m] = boughtAssets[t,m]*(DeltaS[t,m]*(-1)*(-1)**Z[t,m])
                prof = profit[t,m]
                c_array[t+1, m] = c + prof
                gain_array[t+1,m] = c_array[t+1,m] - cash
                if t < (a_max.shape[0]-1):
                    a_max[t+1,m] = np.where(S[t+1,m] == 0, 0, np.divide(c_array[t+1,m], S[t+1,m]))
        gains[etastr][beliefstr] = gain_array.copy()
        assets[etastr][beliefstr] =  boughtAssets.copy()
        assetmaxes[etastr][beliefstr] = a_max.copy()
        cashs[etastr][beliefstr] =  c_array.copy()
        
        #profit = 0*np.ones((T,M))
        #for t in range(T):        
        #    profit[t] = boughtAssets[t]*(DeltaS[t]*(-1)*(-1)**Z[t])

        column_sums = np.sum(profit, axis=0) #calculating and storing values for the graphs
        column_avg = np.mean(column_sums, axis=0)
        column_sd = np.std(column_sums, axis=0)
        individual_gains = profit.flatten()
        key = str(etastr)
        key_list.append(key)
        longkey = f"Switch:{q}-Adverse:{np.round(0.5-ch, decimals=2)}-Omega:{o}-Eta:{e}"
        longkey_list.append(longkey)
        plot_key.append(key)
        #plot_value.append(individual_gains)
        plot_value.append(column_sums)
        value_list.append(column_sums)
        sum_list.append(column_avg)
        ind_list.append(individual_gains)
        sd_list.append(column_sd)
        profits[etastr][beliefstr] = profit
        print(f"loop for omega: {o} and eta: {e}")
    plot_dict = dict(zip(plot_key, plot_value)) # Graphing commands
    plotdata[beliefstr] = plot_value
    plot_dataframe = pd.DataFrame.from_dict(plot_dict, orient='index').transpose()

# create empty dictionary with bot data
column_names = [
'ongoingReturn', 'Rank', 'ResponseId', 'gainAll', 'returnAll','returnAllv2', 'pathseries', 'r',
'stg', 'p', 'a', 'c', 'path', 'portfolio', 'unrealized', 'PlayerName',
'riskRelative', 'risk', 'riskRelativeAvg', 'phase', 'gain', 'phaseReturn',
'wealthALL','wealthAllv2', 'Treatment', 'roundSeries', 'priceSeries', 'assetseries',
'ongoingReturnSeries', 'phaseWealth', 'GainCalc',
'currWealth', 'a_diff', 'type_path']

Scopy = S.copy()

assetsCopy = {}
test = {}
for key, value in assets.items():
    assetsCopy[key] = value.copy()
    test[key] = value.copy()
cashsCopy = {}
for key, value in cashs.items():
    cashsCopy[key] = value.copy()
profitsCopy = profits.copy()

data = create_bots(len(etas),8, column_names)

Ro = T+1
RoT = Ro*11
c = cash

for i in range(1,9):
    series_name = f'{i}'
    serie = i
    data[series_name] = populate_dataframes(data[series_name], T+1, 11, Scopy)
    Scopy = Scopy[:T+1, 11:]
    for i, (key, value) in enumerate(assetsCopy.items()):
        player_name = f'player_{i+1}'
        player_name_data = f'"Player{i+1}"'
        CRRA = f'CRRA_{key}'
        data[series_name][player_name]['PlayerName'] = player_name_data
        data[series_name][player_name]['ResponseId'] = CRRA
        data[series_name][player_name]['pathseries'] = series_name
        data[series_name][player_name]['a'] = value["1.0"][:T+1, :11].reshape(-1, order="F")
        data[series_name][player_name]['a_diff'] = value["1.0"][:T+1, :11].reshape(-1, order="F")
        for i in range(0, RoT, Ro):
            #changes in assets held for further calculations
            batch = data[series_name][player_name]['a'][i:i+Ro]
            
            difference_array = np.diff(batch, axis=0)
            difference_array = np.insert(difference_array, 0, 0, axis=0)
            
            data[series_name][player_name]['a_diff'][i:i+Ro] = difference_array
            data[series_name][player_name]['a_diff'][i:i+1] = data[series_name][player_name]['a'][i:i+1]
            count = int((((i/40)+(int(series_name)-1)*10)+1)+(int(series_name)-1))
            path = f'path{count}'
            pathindex = count-1
            data[series_name][player_name]['path'][i:i+Ro] = path
            data[series_name][player_name]['type_path'][i:i+Ro] = datadump['type'][serie-1]
            
            
        value["1.0"] = value["1.0"][:T+1, 11:]
    for i, (key, value) in enumerate(cashsCopy.items()):
        player_name = f'player_{i+1}'
        data[series_name][player_name]['currWealth'] = value["1.0"][:T+1, :11].reshape(-1, order="F")
        data[series_name][player_name]['c'] = data[series_name][player_name]['currWealth'] - data[series_name][player_name]['a']*data[series_name][player_name]['p']
        data[series_name][player_name]['GainCalc'] = data[series_name][player_name]['currWealth'] - c
        value["1.0"] = value["1.0"][:T+1, 11:]   

        for i in range(0, RoT, Ro):
            #calculations of ongoing return
            batch = data[series_name][player_name]['currWealth'][i:i+Ro]
            ongoingReturn = ((batch/c)-1)*100
            data[series_name][player_name]['ongoingReturn'][i:i+Ro] = np.round(ongoingReturn, decimals=2)
        for i in range(Ro, RoT, Ro):
            #calculations of ongoing return
            if i==Ro :
                last = data[series_name][player_name]['GainCalc'][i-1:i]
                lastCalc = last.iloc[0]
            else:
                last = data[series_name][player_name]['gainAll'][i-1:i]
                lastCalc = last.iloc[0]    
            batch = data[series_name][player_name]['GainCalc'][i:i+Ro]
            gainAll = batch + lastCalc
            data[series_name][player_name]['gainAll'][i:i+Ro] = gainAll
            batch = data[series_name][player_name]['p'][i-Ro:i]
            price_list = batch.tolist()
            price_string = '[{}]'.format(', '.join(map(str, price_list)))
            data[series_name][player_name]['priceSeries'][i:i+Ro] = price_string
            data[series_name][player_name]['priceSeries'][i+1:i+Ro] = np.nan
            
            batch = data[series_name][player_name]['p'][RoT-Ro:RoT]
            price_list = batch.tolist()
            price_string = '[{}]'.format(', '.join(map(str, price_list)))
            data[series_name][player_name]['priceSeries'][RoT-1:RoT] = price_string
            
            batch = data[series_name][player_name]['a'][i-Ro:i]
            asset_list = batch.tolist()
            asset_string = '[{}]'.format(', '.join(map(str, asset_list)))
            data[series_name][player_name]['assetseries'][i:i+Ro] = asset_string
            data[series_name][player_name]['assetseries'][i+1:i+Ro] = np.nan
            
            batch = data[series_name][player_name]['a'][RoT-Ro:RoT]
            asset_list = batch.tolist()
            asset_string = '[{}]'.format(', '.join(map(str, asset_list)))
            data[series_name][player_name]['assetseries'][RoT-1:RoT] = asset_string
            
            batch = data[series_name][player_name]['r'][i-Ro:i]
            round_list = batch.tolist()
            round_string = '[{}]'.format(', '.join(map(str, round_list)))
            data[series_name][player_name]['roundSeries'][i:i+Ro] = round_string
            data[series_name][player_name]['roundSeries'][i+1:i+Ro] = np.nan
            data[series_name][player_name]['roundSeries'][RoT-1:RoT] = round_string
            batch = data[series_name][player_name]['ongoingReturn'][i-Ro:i]
            return_list = batch.tolist()
            return_list = [round(num, 2) for num in return_list]
            return_string = '[{}]'.format(', '.join(map(str, return_list)))
            data[series_name][player_name]['ongoingReturnSeries'][i:i+Ro] = return_string
            data[series_name][player_name]['ongoingReturnSeries'][i+1:i+Ro] = np.nan
            
            batch = data[series_name][player_name]['ongoingReturn'][RoT-Ro:RoT]
            return_list = batch.tolist()
            return_list = [round(num, 2) for num in return_list]
            return_string = '[{}]'.format(', '.join(map(str, return_list)))
            data[series_name][player_name]['ongoingReturnSeries'][RoT-1:RoT] = return_string
            
            lastWealth = data[series_name][player_name]['currWealth'][i-1:i]
            lastWealth_val = lastWealth.iloc[0]
            data[series_name][player_name]['phaseWealth'][i:i+41] = lastWealth_val
            if i==Ro :
                last = data[series_name][player_name]['currWealth'][i-1:i]
                lastCalc = last.iloc[0]
            else:
                last = data[series_name][player_name]['wealthALL'][i-1:i]
                lastCalc = last.iloc[0] 
            batch = data[series_name][player_name]['currWealth'][i:i+Ro]
            gainAll = batch + lastCalc
            data[series_name][player_name]['wealthALL'][i:i+Ro] = gainAll
            data[series_name][player_name]['wealthAllv2'][i:i+Ro] = data[series_name][player_name]['wealthALL'][i:i+Ro]
            if i >= 2 * Ro:
                data[series_name][player_name]['wealthAllv2'][i:i+Ro] = data[series_name][player_name]['wealthALL'][i:i+Ro]-data[series_name][player_name]['c'].iloc[40]
            
            
            lastRet = data[series_name][player_name]['ongoingReturn'][i-1:i]
            lastRet_val = lastRet.iloc[0]
            data[series_name][player_name]['phaseReturn'][i:i+Ro] = lastRet_val
            
            lastGain = data[series_name][player_name]['GainCalc'][i-1:i]
            lastGain_val = lastGain.iloc[0]
            data[series_name][player_name]['gain'][i:i+Ro] = lastGain_val
            
            stage = i/(T+1)
            endow = c*(stage)
            batch = data[series_name][player_name]['wealthALL'][i:i+Ro]
            ret = (((batch-2500)/endow)-1)*100
            if stage == 1:
                print(f"debug stage: {stage}")
                print(f"debug endow: {endow}")
                print(f"debug batch: {batch}")
                print(f"debug return: {ret}")
                print(f"cash value: {data[series_name][player_name]['c'].iloc[40]}")
            data[series_name][player_name]['returnAll'][i:i+Ro] = round_var(ret)
            
            data[series_name][player_name]['returnAllv2'][i:i+Ro] = data[series_name][player_name]['returnAll'][i:i+Ro]
            if i >= 2 * Ro:
                endow = c*(stage-1)
                batch = data[series_name][player_name]['wealthAllv2'][i:i+Ro]
                ret = (((batch-2500)/endow)-1)*100
                data[series_name][player_name]['returnAllv2'][i:i+Ro] = round_var(ret)
            
            
            
        for i in range(0, RoT, Ro):
            portfolio = data[series_name][player_name]['unrealized'][i:i+Ro]
            assets = data[series_name][player_name]['a'][i:i+Ro]
            assetChange = data[series_name][player_name]['a_diff'][i:i+Ro]
            price = data[series_name][player_name]['p'][i:i+Ro]
            for index, value in portfolio.items():
                if index == i:
                   portfolio[index] = price[index]*assetChange[index]
                else:
                    if assetChange[index] < 0:
                        calculated_value = portfolio[index-1] + (portfolio[index-1] / assets[index-1]) * assetChange[index]
                    else:
                        calculated_value = portfolio[index-1] + price[index]*assetChange[index]
                    portfolio[index] = max(calculated_value, 0)
                    if assets[index] == 0:
                        portfolio[index] = 0
            data[series_name][player_name]['unrealized'][i:i+Ro] = round_var(portfolio)
            portfolioReturn = data[series_name][player_name]['portfolio'][i:i+Ro]
            for index, value in portfolio.items():
                if assets[index] != 0:
                    avgP = portfolio[index]/assets[index]
                    ret = price[index]/avgP
                    portfolioReturn[index] = ((ret-1)*100).round(2)
                else:
                    portfolioReturn[index] = 0
              
            #change values in last round of last phase for final screen
            data[series_name][player_name]['phaseWealth'][RoT-1:RoT] = data[series_name][player_name]['currWealth'][RoT-1:RoT]
            data[series_name][player_name]['phaseReturn'][RoT-1:RoT] = data[series_name][player_name]['ongoingReturn'][RoT-1:RoT] 
            
    def extract_stage_number(string):
        match = re.match(r'stage_(\d+)', string)
        if match:
            return int(match.group(1))
        else:
            return None
    def extract_round_number(string):
        match = re.match(r'round_(\d+)', string)
        if match:
            return int(match.group(1))
        else:
            return None
    
    def extract_rank_number(string):
        match = re.match(r'rank_(\d+)', string)
        if match:
            return int(match.group(1))
        else:
            return None
    
    def rank_transfer(rankdf, data):
        for stage in rankdf.keys():
            for rnd in rankdf[stage].keys():
                for rank in rankdf[stage][rnd].keys():
                    rk = extract_rank_number(rank)
                    playerid = rankdf[stage][rnd][rank].iloc[0,2]
                    r = rankdf[stage][rnd][rank].iloc[0,6]
                    s = rankdf[stage][rnd][rank].iloc[0,7]
                    path = rankdf[stage][rnd][rank].iloc[0,11]
                    for player in data.keys():
                        # Define conditions
                        condition = (data[player]['ResponseId'] == playerid) & (data[player]['r'] == r) & (data[player]['stg'] == s) & (data[player]['path'] == path)
                        
                        # Update cell in Column3 based on condition
                        data[player].loc[condition, 'Rank'] = rk                    
    
    
os.chdir(folder_path)                    
for x, (key, value) in enumerate(data.items()):
    series = key               
    data_players = data[key]
    data_rounds = {}
    
    # Sorting/Ranking 0 - ongoingReturn, 1 - Rank, 2 - ResponseID, 3 - gainAll, 4 - returnAll, 5 - returnAllv2, 6 - pathseries, 20-gain
    for i in range(0, data_players["player_1"]["stg"].nunique()) :
        stage = "stage_" + str(i)
        data_rounds[stage] = {}
        for n in range(0, T+1):
              rnd = "round_" + str(n)
              data_rounds[stage][rnd] = {}
              for j in range(0, len(data_players)):
                  player = "player_" + str(j+1)
                  data_rounds[stage][rnd][player] = data_players[player].loc[(data_players[player]["stg"] == i) & \
                  (data_players[player]["r"] == n) & (data_players[player]["phase"] == "regular")]
              for key in  data_rounds[stage][rnd].keys():
                  for other_key in  data_rounds[stage][rnd].keys():
                      if stage == "stage_10" and rnd == "round_40":
                        # Use your custom condition here instead of gain comparison
                        if float(data_rounds[stage][rnd][other_key].iloc[0, 0]) > float(data_rounds[stage][rnd][key].iloc[0, 0]):
                            data_rounds[stage][rnd][key].iloc[0, 1] += 1
                      else:
                        # Default sorting by gain
                        if float(data_rounds[stage][rnd][other_key].iloc[0, 20]) > float(data_rounds[stage][rnd][key].iloc[0, 20]):
                            data_rounds[stage][rnd][key].iloc[0, 1] += 1 
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
        for n in range(0, T+1):
            rnd = "round_" + str(n)
            data_ranked[stage][rnd] = {}
            for j in range(0, len(data_players)):
                rank = "rank_" + str(j+1)
                data_ranked[stage][rnd][rank] = ()
                for key in  data_rounds[stage][rnd].keys():
                    if data_rounds[stage][rnd][key].iloc[0,1] == j+1:
                        data_ranked[stage][rnd][rank] = data_rounds[stage][rnd][key]     
    
    playername_dict = {}
    
    for key in data_ranked["stage_0"]["round_0"]:
        player_number = key.split('_')[1]
        variable_name = 'Player'                     
        new_key = f"Player{player_number}"
        new_value = data_ranked["stage_0"]["round_0"][key]["ResponseId"].values.tolist()
        print(new_value)
        playername_dict[new_key] = new_value
    
    for i in range(0, data_players["player_1"]["stg"].nunique()) :
        stage = "stage_" + str(i)
        for n in range(0, T+1):
            rnd = "round_" + str(n)
            for j in range(0, len(data_players)):
                rank = "rank_" + str(j+1)
                for key in playername_dict:
                    dictValue = playername_dict[key][0]
                    DFValue = data_ranked[stage][rnd][rank].iloc[0,2]
                    if dictValue == DFValue:

                        print(key)
                        playername = f'"{key}"'
                        data_ranked[stage][rnd][rank].iloc[0,15] = playername
                        """ Playername is position 15"""
                    else:
                        print("no match")
    
    """
    dfOutput = pd.DataFrame()
    for i in range(0, data_players["player_1"]["stg"].nunique()) :
        stage = "stage_" + str(i)
        for n in range(0, T):
            rnd = "round_" + str(n)
            for key in  data_rounds[stage][rnd].keys():
                dfOutput = dfOutput.append(data_rounds[stage][rnd][key], ignore_index=True)
                
    dfOutput.to_csv( f"botdata-series{series}.csv", index=False, encoding='utf-8-sig')
    """
    if outputtxt ==1:
        string = "const data = {"
        string += "\n"
        
        for i in range(1, data_players["player_1"]["stg"].nunique()) :
            stage = "stage_" + str(i)
            stagestring = "stage_" + str(i-1)
            string += "\t"
            string += f'{stagestring}: '
            string += '{'
            string += "\n"
            for n in range(0, T+1):
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
        
        def replace_nan(input_string):
            return input_string.replace('nan', 'NaN')
        
        string = replace_nan(string)
        typepath = data_ranked[stage][rnd][rank][key].iloc[-1]
        file_name = f"javastrings_bots_{typepath}.txt"
        subfolder = 'Botdata'
        file_path = os.path.join(subfolder, file_name)
        file_path = os.path.join(folder_path, file_path)
        export_string_to_txt_file(string, file_path)
        
        #rank_transfer(data_ranked, data[str(x+1)])
        min_ongoing_returns = []
        max_ongoing_returns = []
        all_ongoing_returns = []
        
        # Loop through each stage from stage_0 to stage_10
        for i in range(11):
            stage_key = f"stage_{i}"
        
            # Access the 'round_40' dictionary within the current stage
            round_data = data_rounds[stage_key]["round_40"]
        
            # Collect all ongoingReturn values across all 5 players
            ongoing_returns_stage = [round_data[f"player_{j}"]["ongoingReturn"] for j in range(1, len(etas) + 1)]
        
            # Flatten the list (in case ongoingReturn is a list or array)
            ongoing_returns_stage = [val for sublist in ongoing_returns_stage for val in sublist]
        
            # Determine min and max values
            min_return_stage = min(ongoing_returns_stage)
            max_return_stage = max(ongoing_returns_stage)
        
            # Store min, max, and all ongoingReturn values
            min_ongoing_returns.append(min_return_stage)
            max_ongoing_returns.append(max_return_stage)
            all_ongoing_returns.append(ongoing_returns_stage)  # Store everything
        
        # Store the min, max, and all values in the results dictionary
        results[(series)] = {
            "min": min_ongoing_returns,
            "max": max_ongoing_returns,
            "all": all_ongoing_returns  # Contains all values
        }
            
        
            #rank_transfer(data_ranked, data[str(x+1)])
        
        
        
        for key in data.keys():
            for player in data[key].keys():
                risk = data[key][player].iloc[0,2]
                collection["Rank"][risk] = pd.concat([collection["Rank"][risk], data[key][player]["Rank"]], axis=1)
                collection["Gain"][risk] = pd.concat([collection["Gain"][risk], data[key][player]["gainAll"]], axis=1)   
                
        for stage in range(1, 11):
            stage_key = f"stage_{stage}"
            if stage_key not in data_rounds:
                continue
        
            if "round_40" not in data_rounds[stage_key]:
                continue
        
            response_ranks = {}
            response_returns = {}
        
            for player in range(1, len(etas) + 1):
                player_key = f"player_{player}"
                if player_key not in data_rounds[stage_key]["round_40"]:
                    continue
        
                df = data_rounds[stage_key]["round_40"][player_key]
                if "ResponseId" not in df.columns or "Rank" not in df.columns:
                    print(f"Missing required columns in {stage_key}, {player_key}")
                    break
        
                response_id = df.iloc[0]["ResponseId"]
                rank = df.iloc[0]["Rank"]
        
                # Choose the appropriate return column
                if stage == 10:
                    if "returnAllv2" not in df.columns:
                        print(f"Missing 'ReturnAllv2' in stage 10, {player_key}")
                        break
                    returns = df.iloc[0]["returnAllv2"]
                else:
                    returns = df.iloc[0]["ongoingReturn"]
        
                if response_id in response_ranks:
                    print(f"Duplicate ResponseId '{response_id}' found in {stage_key}, {player_key}")
                    break
        
                response_ranks[response_id] = rank
                response_returns[response_id] = returns
        
            # Add ranks
            for response_id, rank in response_ranks.items():
                if response_id not in result_dict:
                    result_dict[response_id] = []
                result_dict[response_id].append(rank)
        
            # Add returns
            for response_id, returns in response_returns.items():
                if response_id not in result_dict_returns:
                    result_dict_returns[response_id] = []
                result_dict_returns[response_id].append(returns)
        
            # Special handling for stage 10
            if stage == 10:
                # Save stage 10 returns to final_results_returns (append per response_id)
                for response_id, return_value in response_returns.items():
                    if response_id not in final_results_returns:
                        final_results_returns[response_id] = []
                    final_results_returns[response_id].append(return_value)
            
                # Compute ranks based on stage 10 returns
                sorted_responses = sorted(response_returns.items(), key=lambda x: x[1], reverse=True)
            
                # Save ranks in final_results (append per response_id)
                for rank, (response_id, _) in enumerate(sorted_responses, start=1):
                    if response_id not in final_results:
                        final_results[response_id] = []
                    final_results[response_id].append(rank)


            
    # Define the bin edges to exactly match the categories 1, 2, 3, 4, 5
    bins = np.arange(0.5, 6.5, 1)  # Centers bins around integer values
    
    # Generate and save histograms for each ResponseId
    for response_id, rank_values in result_dict.items():
        plt.figure(figsize=(8, 5))
        plt.hist(rank_values, bins=bins, edgecolor="black", alpha=0.7, align='mid', rwidth=0.8)
        
        plt.xlabel("Rank Values")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Rank Values for {response_id}")
        plt.xticks([1, 2, 3, 4, 5])  # Ensure x-axis only has valid rank values
        plt.grid(axis="y", alpha=0.75)
        
        # Save the figure
        save_path = os.path.join(folder_path, f"{response_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory   
        
    pkl_path = os.path.join(folder_path, "result_dict.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(result_dict, f)
    
    print(f"result_dict saved successfully to {pkl_path}")
    
    # Generate and save histograms for each ResponseId
    for response_id, return_values in result_dict_returns.items():
        plt.figure(figsize=(8, 5))
        plt.hist(return_values, edgecolor="black", alpha=0.7, align='mid', rwidth=0.8)
        
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Returns for {response_id}")
        plt.grid(axis="y", alpha=0.75)
        
        # Save the figure
        save_path = os.path.join(folder_path, f"{response_id}_returns.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory   
        
    pkl_path = os.path.join(folder_path, "result_dict_returns.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(result_dict_returns, f)
    
    print(f"result_dict saved successfully to {pkl_path}")

    # Histogram of stage 10 returns across iterations
    for response_id, return_values in final_results_returns.items():
        plt.figure(figsize=(8, 5))
        plt.hist(return_values, edgecolor="black", alpha=0.7, align='mid', rwidth=0.8)
    
        plt.xlabel("Stage 10 Return (ReturnAllv2)")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Stage 10 Returns for {response_id}")
        plt.grid(axis="y", alpha=0.75)
    
        save_path = os.path.join(folder_path, f"{response_id}_stage10_returns.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    # Histogram of stage 10 ranks across iterations
    for response_id, rank_values in final_results.items():
        plt.figure(figsize=(8, 5))
        plt.hist(rank_values,bins=bins, edgecolor="black", alpha=0.7, align='mid', rwidth=0.8)
    
        plt.xlabel("Stage 10 Rank")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Stage 10 Ranks for {response_id}")
        plt.grid(axis="y", alpha=0.75)
    
        save_path = os.path.join(folder_path, f"{response_id}_stage10_ranks.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

avgResults = {}

# Variables to store the highest and lowest individual values
highest_individual_value = float('-inf')
lowest_individual_value = float('inf')

# Iterate over each series-itera combination in results to compute the averages
for key, value in results.items():
    # Calculate the average of the min and max lists
    avg_min = np.mean(value["min"])
    avg_max = np.mean(value["max"])
    
    # Update the highest and lowest individual values from the min and max lists
    highest_individual_value = max(highest_individual_value, max(value["min"]), max(value["max"]))
    lowest_individual_value = min(lowest_individual_value, min(value["min"]), min(value["max"]))
    
    # Store the averages in avgResults
    avgResults[key] = {
        "avg_min": avg_min,
        "avg_max": avg_max
    }

# Find the highest and lowest averages across all series-itera combinations
all_avg_min = [entry["avg_min"] for entry in avgResults.values()]
all_avg_max = [entry["avg_max"] for entry in avgResults.values()]

highest_avg_min = max(all_avg_min)
lowest_avg_min = min(all_avg_min)
highest_avg_max = max(all_avg_max)
lowest_avg_max = min(all_avg_max)

# Display the results
print("Average Results per Series-Itera Combination:")
print(avgResults)
print("\nHighest Average Min:", highest_avg_min)
print("Lowest Average Min:", lowest_avg_min)
print("Highest Average Max:", highest_avg_max)
print("Lowest Average Max:", lowest_avg_max)
print("\nHighest Individual Value:", highest_individual_value)
print("Lowest Individual Value:", lowest_individual_value)
    


import matplotlib.pyplot as plt

# Collect all ongoingReturn values from all (series, itera) keys
all_returns = []

for key, value in results.items():  # Loop through all (series, itera) entries
    all_returns.extend([val for sublist in value["all"][1:] for val in sublist])  # Skip first list  # Flatten and collect

transformed_returns = [((val / 100) + 1) * 2500 / 10 for val in all_returns]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original histogram
axes[0].hist(all_returns, bins=30, edgecolor="black", alpha=0.75)
axes[0].set_xlabel("Ongoing Return Values")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Histogram of Returns")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot the transformed histogram
axes[1].hist(transformed_returns, bins=30, edgecolor="black", alpha=0.75, color="orange")
axes[1].set_xlabel("Transformed Values")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Histogram of Bonus in cents")
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

# Show the plots
plt.tight_layout()
plt.show()

print("Highest Return:",  max(all_returns))
print("Lowest Return:",  min(all_returns))
import statistics
print("Avg Return:",  statistics.mean(all_returns))
print("Median Return:",  statistics.median(all_returns))



### Extracting data for R estimation
# Extract and convert to list
values = beliefs["1.0"][0:40, 131].tolist()

processed = [round(v, 4) for v in values]

# Convert to JS array string
js_array = '[' + ', '.join(map(str, processed)) + ']'

print(js_array)

filtered_data = data["12"]["player_5"][data["12"]["player_5"]['r'] != 40]

# Extract values from 'currWealth' column (you can also slice it further if needed)
values = filtered_data['currWealth'].tolist()

# Convert to JS array string
js_array = '[' + ', '.join(map(str, values)) + ']'

print(js_array)

filtered_data = data["12"]["player_5"][data["12"]["player_5"]['r'] != 40]

# Extract values from 'currWealth' column (you can also slice it further if needed)
values = filtered_data['a'].tolist()

# Convert to JS array string
js_array = '[' + ', '.join(map(str, values)) + ']'

print(js_array)

filtered_data = data["12"]["player_5"][data["12"]["player_5"]['r'] != 40]

# Extract values from 'currWealth' column (you can also slice it further if needed)
values = filtered_data['p'].tolist()

# Convert to JS array string
js_array = '[' + ', '.join(map(str, values)) + ']'

print(js_array)