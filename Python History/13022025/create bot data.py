# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:14:44 2024

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

#folder_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game"
folder_path = r"D:\Surfdrive\Projects\Copy Trading\Trading game"


collection = {}
collection["Rank"] = {}
collection["Gain"] = {}
for key in collection.keys():
    collection[key]["CRRA_-1.5"] = pd.DataFrame()
    collection[key]["CRRA_0"] = pd.DataFrame()
    collection[key]["CRRA_1"] = pd.DataFrame()
    collection[key]["CRRA_3"] = pd.DataFrame()
    collection[key]["CRRA_6"] = pd.DataFrame()

for itera in range(1,3): 
    print(itera)
    ## Setting up the simulation
    #for q in [0.1, 0.2, 0.3, 0.4]:
    for q in [0.15]:
        #for ch in [0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
        for ch in [0.15]:
            T = 40 #number of periods
            M = 10000 #number of iterations to run
            q = q #probability to switch states
            ch = ch #governs the chance to get the opposite price change per state, e.g. down in good state, Pr = 0.5-ch
            S0 = 250 #initial stock price
            c = 2500 #cash endowment
            cash = c #for calculation of gain array
            ks = [5, 10, 15] # Define the possible outcomes
            probabilities = [1/3, 1/3, 1/3] # Set the probabilities for each outcome (uniform distribution)
            
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
            
            
            # Find the indices of the first occurrence of 0 in each column
            zero_indices = (S == 0).argmax(axis=0)
    
            # Iterate over each column
            for j in range(S.shape[1]):
                if j < len(zero_indices):
                    first_zero_index = zero_indices[j]
                else:
                    # Handle the case where j is out of bounds
                    print("Index out of bounds!")    
                if first_zero_index != 0 and (S[:, j] == 0).any():  # If there's at least one zero in the column
                    S[first_zero_index:, j] = 0
            diff_array = S[-1, :] - S[0, :]
            deciles = np.percentile(diff_array, np.arange(0, 110, 10))
            high_indices = np.where(diff_array > deciles[8])[0]
            low_indices = np.where(diff_array < deciles[2])[0]
            for i in range (1,11):
                """
                Output txt documents of lists of picked paths zeroes/ones/low/top and final
                Save all paths to excel and pickle
                """
                #picking paths randomly, 5 starting in the good state, 5 in the bad
                # Find indices of 0s and 1s
                zero_indices = np.where(theta0 == 0)[0]
                one_indices = np.where(theta0 == 1)[0]
                # Randomly select 5 indices with 0s and 5 indices with 1s
                selected_zero_indices = np.random.choice(zero_indices, 5, replace=False)
                selected_one_indices = np.random.choice(one_indices, 5, replace=False)        
                # Store both sets of indices in one list
                selected_indices = np.concatenate([selected_zero_indices, selected_one_indices])        
                # Shuffle the selected indices
                np.random.shuffle(selected_indices)
                if i == 1:
                    final_selection = selected_indices
                    final_zeroes = selected_zero_indices
                    final_ones = selected_one_indices
                else:
                    final_selection = np.concatenate((final_selection, selected_indices))
                    final_zeroes = np.concatenate((final_zeroes, selected_zero_indices))
                    final_ones = np.concatenate((final_ones, selected_one_indices))
            
            # Randomly select one additional column index
            additional_index = np.random.randint(S.shape[1])
            # Randomly pick one that results in top 20% 
            high_additional_index = np.random.choice(high_indices, 1, replace=False)
            # Randomly pick one that results in bottom 20% 
            low_additional_index = np.random.choice(low_indices, 1, replace=False)
            # Ensure additional_index is unique
            while high_additional_index in final_selection:
                high_additional_index = np.random.choice(high_indices, 1, replace=False)
            additional_index =  high_additional_index
            # Insert the additional index at the start of the list
            final_selection = np.insert(final_selection, 0, additional_index, axis=0)
            final_selection = np.insert(final_selection, 11, additional_index, axis=0)
            final_selection = np.insert(final_selection, 22, additional_index, axis=0)
            final_selection = np.insert(final_selection, 33, additional_index, axis=0)
            final_selection = np.insert(final_selection, 44, additional_index, axis=0)
            
            while low_additional_index in final_selection:
                low_additional_index = np.random.choice(low_indices, 1, replace=False)
            additional_index =  low_additional_index
            # Insert the additional index at the start of the list
            final_selection = np.insert(final_selection, 55, additional_index, axis=0)
            final_selection = np.insert(final_selection, 66, additional_index, axis=0)
            final_selection = np.insert(final_selection, 77, additional_index, axis=0)
            final_selection = np.insert(final_selection, 88, additional_index, axis=0)
            final_selection = np.insert(final_selection, 99, additional_index, axis=0)
            
            # Keep only the columns from S that correspond to the selected indices
            selected_S = S[:, final_selection]
            selected_zeroes = S[:, final_zeroes]
            selected_ones = S[:, final_ones]
            
            # Write selected price paths to txt in JS format for hardcoded paths
            stringp = ""

            # Loop over the columns
            for col in range(selected_S.shape[1]):
                pathname = f"pricePaths.path{col+1}"
                stringp += f"{pathname} = ["
                # Loop over the rows
                for row in range(selected_S.shape[0]):
                    #print(selected_S[row, col])
                    stringp += f"{selected_S[row, col]},"
                stringp += "];"
                stringp += "\n" 

            def export_string_to_txt_file(string, file_name):
                with open(file_name, "w") as file:
                    file.write(stringp)
                    
            file_name = f"javastrings_price_series_{itera}.txt"
            subfolder = 'Paths'
            file_path = os.path.join(subfolder, file_name)
            file_path = os.path.join(folder_path, file_path)
            export_string_to_txt_file(stringp, file_path)
            
            # Create data dump for all paths, zero/one selected indices,starting indices, finals selection
            datadump = {}
            datadump["all_generated_paths"] = S
            datadump["zeroes_paths"] = selected_zeroes
            datadump["ones_paths"] = selected_ones
            datadump["zeroes_indeces"] = final_zeroes
            datadump["ones_indeces"] = final_ones
            datadump["low_index"] = low_additional_index
            datadump["high_index"] = high_additional_index
            datadump["selected_paths"] = selected_S
            datadump["selected_indeces"] = final_selection
            datadump["good_bad_binary"] = np.array([1 if a in final_ones else 0 for a in final_selection])

            
            file_name = f"data_{itera}.pk"
            subfolder = 'Data'
            file_path = os.path.join(subfolder, file_name)
            file_path = os.path.join(folder_path, file_path)
            
            with open(file_path, 'wb') as file:
                pickle.dump(datadump, file)
            
            """
            # Removimg extreme price paths
            def filter_columns(array, percentile_low=1, percentile_high=99):
                final_row = array[-1]
                
                #Determine percentile cutoffs
                low_cutoff = np.percentile(final_row, percentile_low)
                high_cutoff = np.percentile(final_row, percentile_high)
                
                #Create boolean masks for filtering
                mask = (final_row >= low_cutoff) & (final_row <= high_cutoff)
                
                #Filter columns using boolean indexing
                filtered_array = array[:, mask]
                
                return filtered_array
            
            S_filtered = filter_columns(S, percentile_low=20, percentile_high=80)
            """ 
            
            """ Done differently now above
            #function to randomly pick paths from all generated ones
            def pick_random_paths(array, num_columns=10):
                # Get the number of columns in the array
                num_total_columns = array.shape[1]
                
                # Ensure num_columns is not greater than the total number of columns
                num_columns = min(num_columns, num_total_columns)
                
                # Randomly pick column indices
                selected_columns = np.random.choice(num_total_columns, size=num_columns, replace=False)
                
                # Create a new array with selected columns
                new_array = array[:, selected_columns]
                
                return new_array
            """
            
            
            file_path = os.path.join(folder_path, f"all_paths.csv")
            np.savetxt(file_path, S, delimiter=',', fmt='%d')
            
            S = selected_S #Overwriting original price array
            #S = pick_random_paths(S,101)
            
            file_path = os.path.join(folder_path, f"picked_paths.csv")
            np.savetxt(file_path, S, delimiter=',', fmt='%d')
            
            T, M = S.shape #adjusting dimension variables
            T = T-1   
            
            #Creating new Z and DeltaS arrays that are needed for profit and belief calculation, based on trimmed price array S
            def compute_Z_new(S):
                # Compute the differences between consecutive rows
                differences = np.diff(S, axis=0)
                
                # Create Zn based on the sign of the differences
                Zn = np.where(differences > 0, 1, 0)
                
                
                return Zn
            
            Z = compute_Z_new(S)
            
            def compute_DeltaS_new(S):
                # Compute the differences between consecutive rows
                differences = np.diff(S, axis=0)
                
                return differences
            
            DeltaS = np.abs(compute_DeltaS_new(S))
            
            
            
             
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
    
    
            beliefs = {}
            if skill == 0:
                for o in omega:
                
                    variable_name = str(o)
                    g = 1
                    belief = 0.5*np.ones((T+1,M))
                    for t in range(T):
                        belief[t+1] = p_update(belief[t], Z[t], o, g)
                    beliefs[variable_name] = belief
            else:
                for g in gamma:
                
                    variable_name = str(g)
                    o = 1.0
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
            if skill == 0:
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
            else:
                for g in gamma:
                    string = str(g)
                    belief = beliefs[string]
                    beliefstr = str(g)
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
                                profit[t,m] = boughtAssets[t,m]*(S[t+1,m]-S[t,m])
                                prof = profit[t, m]
                                c_array[t+1, m] = c + prof
                                gain_array[t+1, m] = c_array[t+1,m] - cash
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
                        longkey = f"Switch:{q}-Adverse:{np.round(0.5-ch, decimals=2)}-Gamma:{g}-Eta:{e}"
                        longkey_list.append(longkey)
                        plot_key.append(key)
                        #plot_value.append(individual_gains)
                        plot_value.append(column_sums)
                        value_list.append(column_sums)
                        sum_list.append(column_avg)
                        ind_list.append(individual_gains)
                        sd_list.append(column_sd)
                        profits[etastr][beliefstr] = profit
                        print(f"loop for gamma: {g} and eta: {e}")
                        
    
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
        
    
    # create empty dictionary with bot data
    column_names = [
    'ongoingReturn', 'Rank', 'ResponseId', 'gainAll', 'returnAll', 'pathseries', 'r',
    'stg', 'p', 'a', 'c', 'path', 'portfolio', 'unrealized', 'PlayerName',
    'riskRelative', 'risk', 'riskRelativeAvg', 'phase', 'gain', 'phaseReturn',
    'wealthALL', 'Treatment', 'roundSeries', 'priceSeries', 'assetseries',
    'ongoingReturnSeries', 'phaseWealth', 'GainCalc',
    'currWealth', 'a_diff', 'good']
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
    
    data = create_bots(5,10, column_names)
    
    Ro = T+1
    RoT = Ro*11
    c = cash
    
    for i in range(1,11):
        series_name = f'{i}'
        serie = i
        data[series_name] = populate_dataframes(data[series_name], T+1, 11, Scopy)
        Scopy = Scopy[:T+1, 11:]
        for i, (key, value) in enumerate(assetsCopy.items()):
            player_name = f'player_{i+1}'
            player_name_data = f'Player{i+1}'
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
                data[series_name][player_name]['good'][i:i+Ro] = datadump['good_bad_binary'][pathindex]
                
                
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
                data[series_name][player_name]['assetseries'][i:i+Ro] = np.nan
                
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
                    
                data[series_name][player_name]['returnAll'][i:i+Ro] = round_var(ret)
                
                
                
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
        
        # Sorting/Ranking 0 - ongoingReturn, 1 - Rank, 2 - ResponseID, 3 - gainAll, 4 - returnAll, 5 - pathseries, 19-gain
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
                          if float(data_rounds[stage][rnd][other_key].iloc[0, 19]) > float(data_rounds[stage][rnd][key].iloc[0, 19]):
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
                            data_ranked[stage][rnd][rank].iloc[0,14] = playername
                            """ Playername is position 14"""
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
            file_name = f"javastrings_bots_series_{series}_{itera}.txt"
            subfolder = 'Botdata'
            file_path = os.path.join(subfolder, file_name)
            file_path = os.path.join(folder_path, file_path)
            export_string_to_txt_file(string, file_path)
        
    
        #rank_transfer(data_ranked, data[str(x+1)])
    
    
    for key in data.keys():
        for player in data[key].keys():
            risk = data[key][player].iloc[0,2]
            collection["Rank"][risk] = pd.concat([collection["Rank"][risk], data[key][player]["Rank"]], axis=1)
            collection["Gain"][risk] = pd.concat([collection["Gain"][risk], data[key][player]["gainAll"]], axis=1)   

plotdata = {}
for row_index in [79, 239, 439]:
    plotdata[str(row_index)] = []
    for risk in ["CRRA_-1.5","CRRA_0","CRRA_1","CRRA_3","CRRA_6",]:
        temp = collection["Gain"][risk].iloc[row_index, :].to_numpy()
        temp = temp.astype('float64')
        plotdata[str(row_index)].append(temp)

#inclusive plot, skills grouped by risk attitude
prch = np.round(0.5-ch, decimals=2)
ticks = [str(x) for x in etas]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
plt.figure(figsize=(14, 10))

positions = np.arange(len(plotdata[next(iter(plotdata))])) * len(plotdata)
width = 0.6

colours = [ "#e0f3db","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#0868ac","#084081"]
colour_dict = {}
#for i, (key, data) in enumerate(plotdata.items()):
#    colour_dict[key] = colours[i]

    
# Calculate means for each set of bars
means = {}
medians = {}
sds = {}
deviation_dict = {}

for i, (key, data) in enumerate(plotdata.items()):
    bp = plt.boxplot(data, positions=positions + i * width, sym='', widths=width)

    color = colours[i]
    set_box_color(bp, color)
    
    plt.plot([], c=colours[i], label=f'Skill: {key}')
   
    
    # Calculate mean and add text annotation at the top
    means[key] = [np.mean(d) for d in data]
    medians[key] = [np.median(d) for d in data]
    sds[key] = [np.std(d) for d in data]

i = 0    
for key in plotdata.keys():

    #print(key)
    for j in range(len(plotdata[key])):
        #print(j)
        #print(positions[i])
        plt.text(positions[j] + (i)* width, medians[key][j], f"{means[key][j]:.2f}", ha='center', va='bottom', color='red', rotation=90, fontsize=8)
        plt.text(positions[j] + (i)* width, medians[key][j], f"{medians[key][j]:.2f}", ha='center', va='top', color='blue', rotation=90, fontsize=8)
        plt.text(positions[j] + (i)* width, means[key][j]+ (2*sds[key][j]), f"{sds[key][j]:.2f}", ha='center', va='bottom', color='green', rotation=90, fontsize=8)
        
    i +=1
    #print(i)
    
plt.title(f"Boxplots across rounds and risk attitudes. State Switching p: {q}, adverse effect p:{prch}, rounds:{T}")
plt.xlabel('CRRA Coefficients')
plt.ylabel('Sum gains/losses')

legend_labels = [f'Round: {key}' for key in plotdata.keys()] + ['Mean', 'Median', 'Standard Deviation']
legend_handles = [plt.Line2D([0], [0], color=colours[i], lw=2) for i in range(len(plotdata))] + [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)
]
plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

plt.xticks(positions + 2 * width, ticks)


plt.tight_layout()
#plt.savefig('boxcompare.png')
#folder_path = r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\CRRA\MonteCarlo\Suggestion"
file_path = os.path.join(folder_path, f"boxplot_sum_switch-{q}_adverse-{prch}_rounds-{T}_decisions-{M*T}_cash-{cash}-diff_crra.png")
fig1 = plt.gcf()
plt.pause(0.1)
fig1.savefig(file_path)