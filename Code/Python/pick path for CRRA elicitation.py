# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:30:19 2025

@author: benja
"""

import numpy as np
import scipy.stats as stats

#Set the seed to make random generation reproducible
seed = 3062024
np.random.seed(seed)
combinations = 100000 #amount of pathseries to created
paths = 10000000 #amount of different paths to be created
# picked 100000 in the current iteration

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

## Setting up the simulation
#for q in [0.1, 0.2, 0.3, 0.4]:
for q in [0.15]:
    #for ch in [0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
    for ch in [0.15]:
        T = 50 #number of periods
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
    
belief_up = ((8*belief*ch)+2-(4*ch))/(4)

# Calculate statistics per column
max_values = np.max(belief_up, axis=0)
min_values = np.min(belief_up, axis=0)
variances = np.var(belief_up, axis=0)
averages = np.mean(belief_up, axis=0)
ranges = max_values - min_values  # Difference between max and min

# Stack all results into a new array
stats_array = np.vstack([max_values, min_values, variances, averages, ranges])

# Row labels for clarity
row_labels = ['Max', 'Min', 'Variance', 'Average', 'Range (Max - Min)']

# Display the statistics
for label, row in zip(row_labels, stats_array):
    print(f"{label}: {row}")
    
    
# compute the mean of each row
mean_of_each = stats_array.mean(axis=1)

# print
for label, m in zip(row_labels, mean_of_each):
    print(f"Mean of {label} values: {m}")

# Find column index where range is the largest
max_range_col = np.argmax(ranges)
print(f"\nColumn with the largest range: {max_range_col}")

# Show all values in that column
column_values = belief_up[:, max_range_col]
print(f"All values in column {max_range_col}: {column_values}")

column_values = stats_array[:, max_range_col]
print(f"All values in column {max_range_col}: {column_values}")

# Define tolerance and target
target = 0.5
tolerance = 0.00025

# Ensure data has at least 10 rows
if belief_up.shape[0] > 9:
    row_9 = belief_up[9]
    mask = np.abs(row_9 - target) <= tolerance
    match_indices = np.where(mask)[0]

    if match_indices.size > 0:
        print(f"Values within ±{tolerance} of {target} found at column(s): {match_indices}")
        print(f"Matching values: {row_9[match_indices]}")
        print(f"at column: {match_indices}")
    else:
        print(f"No values within ±{tolerance} of {target} found in row 9.")
else:
    print("Data has fewer than 10 rows; row index 9 does not exist.")


# Step 2: Identify columns in row 9 where value ∈ [0.45, 0.55]
target = 0.5
tolerance = 0.00025
row_9 = belief_up[9]
matching_mask = np.abs(row_9 - target) <= tolerance
matching_cols = np.where(matching_mask)[0]    
 

# compute the mean of each row
mean_of_each = stats_array.mean(axis=1)

# print
for label, m in zip(row_labels, mean_of_each):
    print(f"Mean of {label} values: {m}")
   
# Step 3: Among matching columns, pick the one with largest variance
if matching_cols.size > 0:
    matching_variances = variances[matching_cols]
    max_var_idx = np.argmax(matching_variances)
    selected_col = matching_cols[max_var_idx]

    # Output the result
    #print(f"Columns with value within ±0.05 of 0.5 in row 9: {matching_cols}")
    print(f"Column {selected_col} has the largest variance ({variances[selected_col]:.4f}) among them.")
    print("Statistics for this column:")
    print(f"  Max:     {max_values[selected_col]}")
    print(f"  Min:     {min_values[selected_col]}")
    print(f"  Variance:{variances[selected_col]}")
    print(f"  Average: {averages[selected_col]}")
    print(f"  Range:   {ranges[selected_col]}")
else:
    print("No columns in row 9 contain values within ±0.05 of 0.5.")
    
if S.shape[1] > selected_col:
    selected_S_column_list = S[:, selected_col].tolist()
    print(f"\nColumn {selected_col} from array 'S' as list:")
    print(selected_S_column_list)
else:
    print(f"Array 'S' does not have column index {selected_col}.")
    