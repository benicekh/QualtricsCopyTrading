# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:44:21 2024

@author: benja
"""
from math import log
import numpy as np

ks = [5, 10, 15]
ch = 0.15

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

p_list = [i * 0.01 for i in range(101)]
eta_list = [-1.5,0,1,3,6]
#print(p_list)

print(assetsBought(2500, 0.45, -1.5, 10))

threshold = np.zeros((len(p_list), len(eta_list)+1))

# Calculate the values for each combination of p and eta
for i, p in enumerate(p_list):
    for j, eta in enumerate(eta_list):
        threshold[i, j] = assetsBought(2500, p, eta, 10)
    threshold[i, len(eta_list)] = p