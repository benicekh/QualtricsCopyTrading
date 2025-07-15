# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 11:45:34 2025

@author: U338144
"""

ks = [5, 10, 15] # Define the possible outcomes
probabilities = [1/3, 1/3, 1/3] # Set the probabilities for each outcome (uniform distribution)
ch = 0.15
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import log
import os
import pandas as pd
import re
import pickle    

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

def plot_assets_vs_transformed_p(c=2500, a_max=10, etas=[-1.5, 0, 1, 3, 6]):
    ps = np.linspace(0.20, 0.80, 100)
    # transform p for display only
    p_disp = ((8 * ps * ch) + 2 - (4 * ch)) / 4

    plt.figure(figsize=(8, 5))
    for eta in etas:
        assets = [assetsBought(c, p, eta, a_max) for p in ps]
        plt.plot(p_disp, assets, label=f'η = {eta}')
    plt.xlabel('P price up')
    plt.ylabel('Assets Bought')
    plt.title(f'Optimal Assets vs P price up')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_assets_vs_transformed_p()
    
def plot_assets_with_dual_x(c=2500, a_max=10, etas=[-1.5, 0, 1, 3, 6]):
    # original p and transformed p
    ps = np.linspace(0.40, 0.75, 100)
    p_disp = ((8 * ps * ch) + 2 - (4 * ch)) / 4

    fig, ax = plt.subplots(figsize=(8, 5))
    for eta in etas:
        assets = [assetsBought(c, p, eta, a_max) for p in ps]
        ax.plot(p_disp, assets, label=f'η = {eta}')

    # Primary x-axis: transformed p
    ax.set_xlabel('P price up')
    ax.set_ylabel('Assets Bought')
    ax.set_title(f'Optimal Assets vs P good state and p price up')
    ax.grid(True)

    # Secondary x-axis on top: original p
    # forward: from transformed back to original
    inv = lambda x: (x - 0.5 + ch) / (2 * ch)
    # forward function for transformed axis
    fwd = lambda x: (2 * ch * x + 0.5 - ch)
    secax = ax.secondary_xaxis('top', functions=(inv, fwd))
    secax.set_xlabel('P good state')

    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_assets_with_dual_x()