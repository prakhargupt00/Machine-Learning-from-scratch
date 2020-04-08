# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:11:51 2019

@author: Prakhar
"""

import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt
import math

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

#sepearting x values fromm training
x_values = train_data[0:30411,0:2] 

#genrating column of ones , for taking care of multiplying w0 to feauture matrix
first = np.ones((30411,1))

#adding column of ones to x_values ,this is for w0  
x_values = np.concatenate([first,x_values],axis=1)

# transpose of feature matrix
x_transpose = x_values.transpose()

#y values 
y_values = train_data[0:30411,2]

#multiplication of x and x transpose
x_cross_x_transpose = np.dot(x_transpose,x_values)

#inverse of multiplication of x and x transpose 
mul_inverse = np.linalg.inv(x_cross_x_transpose)

# multiplication of inverse and x trnaspose
intermediate = np.dot(mul_inverse,x_transpose)

#final weight matrix
weights = np.dot(intermediate,y_values)

#calculating mean for genrating r-squared 
mean = np.mean(test_data[0:,2], dtype=np.float64) 

res_sum=0      # residual sum        
tot_sum=0      #total sum 
y_pred = []
y_true = []

#calulating r squared matrix 
for i in range(len(test_data)):
    predicted_value = weights[0] + weights[1]*test_data[i][0] + weights[2]*test_data[i][1]
    y_pred.append(predicted_value)
    y_true.append(test_data[i][2])
    p=  test_data[i][2]  - predicted_value  
    q = test_data[i][2]  - mean
    res_sum+=p**2
    tot_sum+=q**2
    
print("r2  is : " , 1-res_sum/tot_sum)
print("\n")


#calculating RMSE value 
rmse =0 
for i in range(len(y_pred)):
    rmse += pow((y_pred[i] - y_true[i]),2)

rmse/=len(y_pred) 

rmse = math.sqrt(rmse) 
print("RMSE value is : " , rmse)

