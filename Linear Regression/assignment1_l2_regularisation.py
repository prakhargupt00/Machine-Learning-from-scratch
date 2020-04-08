# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:31:44 2019

@author: acer
"""

#GRADIENT DESCENT ALGORITHM WITH REGULARISATION
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

#data normalisation
data[0:,1]= (data[0:,1] -np.mean(data[0:,1], dtype=np.float64)) / np.std(data[0:,1], dtype=np.float64)
data[0:,2]= (data[0:,2] -np.mean(data[0:,2], dtype=np.float64)) / np.std(data[0:,2], dtype=np.float64)
data[0:,3]= (data[0:,3] -np.mean(data[0:,3], dtype=np.float64)) / np.std(data[0:,3], dtype=np.float64)

train_data = data[0:304411,1:4]  #ignoring first column
test_data  = data[304411: ,1:4]  # 0 to 304411 is 70% of data 

#Lamda
lamda = 10 

# no of iteration 
k = 10

#learning rate
alpha = 1e-6

#losses 
losses = []

#cost or error initially
J=0

#initialise weights
w= np.zeros(3)
    
#calculating the error function using gradient descent with l2 norm regularisation 
for i in range(k):   #num_of_iterations
    h=0     #cost for each iteration
    c0=0
    c1=0
    c2=0
    for j in range(len(train_data)):
        e= w[0] + w[1]*train_data[j][0] + w[2]*train_data[j][1]- train_data[j][2] 
        h+=e**2
        #c0,c1,c2 are used in weight updates
        c0+=e
        c1+=e*train_data[j][0]
        c2+=e*train_data[j][1]
    #adding regularisation to the cost
    for j in range(3):
        h+=lamda*w[j]*w[j]
    h=0.5*h
    J=h    #updating the final cost after each iteration
    #updating wieghts
    w[0]=w[0]-alpha*(c0) - alpha*lamda*abs(w[0])
    w[1]=w[1]-alpha*(c1) - alpha*lamda*abs(w[1])
    w[2]=w[2]-alpha*(c2) - alpha*lamda*abs(w[2])
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

print("\n")    
print("r2  is : " , 1-res_sum/tot_sum)

#calculating RMSE value 
rmse =0 
for i in range(len(y_pred)):
    rmse += pow((y_pred[i] - y_true[i]),2)
losses.append(rmse*0.5)
rmse/=len(y_pred) 

rmse = math.sqrt(rmse) 
print("RMSE value is : " , rmse)

#plt.savefig("l2.jpg")
