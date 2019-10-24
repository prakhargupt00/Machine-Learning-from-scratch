# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 20:50:43 2019

@author: acer
"""
#STOCHAISTIC GRADIENT DESCENT ALGORITHM  
import numpy as np 
import pandas as pd  
import matplotlib.pyplot as pyt

#getting the data from the file 
filename = "3D_spatial_network.csv"
df = pd.read_csv(filename) 

#converting the data into numpy array and splitting into test and training data 
data = df.values
np.random.shuffle(data)

# data normalisation
data[0:,1]= (data[0:,1] -np.mean(data[0:,1], dtype=np.float64)) / np.std(data[0:,1], dtype=np.float64)
data[0:,2]= (data[0:,2] -np.mean(data[0:,2], dtype=np.float64)) / np.std(data[0:,2], dtype=np.float64)
data[0:,3]= (data[0:,3] -np.mean(data[0:,3], dtype=np.float64)) / np.std(data[0:,3], dtype=np.float64)

train_data = data[0:304411,1:4]  #ignoring first column
test_data  = data[304411: ,1:4]  # 0 to 304411 is 70% of data

w= np.zeros(3)

#learning rate
alpha = 1e-5

# no of iteration 
k = 15

#cost
J=0 
    
#calculating the error function using gradient descent 
for i in range(k):   #num_of_iterations
    np.random.shuffle(train_data)
    h=0
    c0=0
    c1=0
    c2=0
    for j in range(len(train_data)):
        e  = w[0] + w[1]*train_data[j][0] + w[2]*train_data[j][1]- train_data[j][2] 
        h += e**2
        #updating wieghts
        w[0]=w[0] - alpha*e
        w[1]=w[1] - alpha*e*train_data[j][0]
        w[2]=w[2] - alpha*e*train_data[j][1]
    h=0.5*h
    J=h
    print(h)

#calculating mean for genrating r-squared 
mean = np.mean(test_data[0:,2], dtype=np.float64) 

res_sum=0      # residual sum        
tot_sum=0      #total sum 
y_pred = []
y_true = []

for i in range(len(test_data)):
    predicted_value = w[0] + w[1]*test_data[i][0] + w[2]*test_data[i][1]
    y_pred.append(predicted_value)
    y_true.append(test_data[i][2])
    p=  test_data[i][2]  - predicted_value  
    q = test_data[i][2]  - mean
    res_sum+=p**2
    tot_sum+=q**2
    
print("r2 error is : " , 1-res_sum/tot_sum)
print("\n")

