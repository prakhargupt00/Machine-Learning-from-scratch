# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:07:00 2019

@author: Prakhar
"""

import random
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

def flip_coin(u):
    """
    u is probability of Head
    1 stands for Head, 0 for Tail
    """
    if random.random() < u:
        return 1
    else:
        return 0
    
def plot_pdf(a, b):
    x = np.linspace(0, 1, 100) 
    y = [beta(x, a, b) for x in x]
    plt.plot(x, y)
    return y
    
def beta(u, a, b):
    ans = ((u**(a-1)) * ((1-u)**(b-1)) * gamma(a+b)) / (gamma(a) * gamma(b))
    return ans
    
flips = [flip_coin(0.8) for i in range(160)]         #choosing 0.7 so that u of maximum likelihood

a = 2
b = 3

plt.figure(1,figsize=(10,10))

x = 0  
for flip in flips:
    if flip == 1:
        a += 1
    else:
       b += 1
    if(x%20==0):
        plot_pdf(a, b)
    x += 1


#for part A    
#plt.savefig("partA.jpg")

#for partB
plt.figure(2,figsize=(10,10))
plot_pdf(a, b)

#plt.savefig("partB.jpg")


