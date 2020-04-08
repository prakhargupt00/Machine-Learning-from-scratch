# -*- coding: utf-8 -*-
"""
# FODS ASSIGNMENT 2
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import math

df = pd.read_csv("3D_spatial_network.csv")

data = df.values
np.random.shuffle(data)

#for degree 4
data =  np.append(data,(data[0:,1]**2).reshape(-1,1),axis=1) 
data = np.append(data,(data[0:,2]**2).reshape(-1,1),axis=1)
data = np.append(data,(data[0:,1]*data[0:,2]).reshape(-1,1),axis=1) 

data = np.append(data,(data[0:,1]**3).reshape(-1,1),axis=1)
data = np.append(data,(data[0:,2]**3).reshape(-1,1),axis=1)
data = np.append(data,((data[0:,1]**2)*data[0:,2]).reshape(-1,1),axis=1)
data = np.append(data,((data[0:,1])*(data[0:,2]**2)).reshape(-1,1),axis=1)

data = np.append(data,(data[0:,1]**4).reshape(-1,1),axis=1)
data = np.append(data,(data[0:,2]**4).reshape(-1,1),axis=1)
data = np.append(data,((data[0:,1]**2)*(data[0:,2]**2)).reshape(-1,1),axis=1)
data = np.append(data,((data[0:,1])*(data[0:,2]**3)).reshape(-1,1),axis=1)
data = np.append(data,((data[0:,1]**3)*(data[0:,2])).reshape(-1,1),axis=1)

# data[0:,1].reshape(-1,1).shape for making 1d array to 2d array

#putting y at last column
out = data[:,3].reshape(-1,1)
data = np.delete(data,3,1)
data = np.append(data,out,axis=1)

# data normalisation

for i in range(len(data[0])):
  data[0:,i]= (data[0:,i] -np.mean(data[0:,i], dtype=np.float64)) / np.std(data[0:,i], dtype=np.float64)

# data[0:,2]= (data[0:,2] -np.mean(data[0:,2], dtype=np.float64)) / np.std(data[0:,2], dtype=np.float64)
# data[0:,3]= (data[0:,3] -np.mean(data[0:,3], dtype=np.float64)) / np.std(data[0:,3], dtype=np.float64)

train_data = data[0:304411,1:len(data[0])]  #ignoring first column
test_data  = data[304411: ,1:len(data[0])]  # 0 to 304411 is 70% of data


#initialise weights
w= np.zeros(len(train_data[0]))

#learning rate
eta = 1e-7

# no of iteration
k = 1000

loss = []    
epoch = []

#calculating the error function using gradient descent 
for i in range(k):
  h=0
  c=np.zeros(len(w))
  
  for j in range(len(train_data)):
    e=w[0] 
    for m in range(len(w)-1):
      e+=w[m+1]*train_data[j][m]
    e-=train_data[j][len(w)-1]
    h+=e**2
    #ci are used in weight updates
    c[0]=e
    for m in range(len(c)-1):
      c[m+1]+=e*train_data[j][m]
  h=0.5*h
  loss.append(h)
  epoch.append(i)
  #updating wieghts
  for j in range(len(w)):
    w[j]=w[j]-eta*c[j] 
  
  print(h)

#calculating mean for generating r-squared 
mean = np.mean(test_data[0:,len(w)-1], dtype=np.float64) 

res_sum=0      # residual sum        
tot_sum=0      #total sum 
y_pred = []
y_true = []

for i in range(len(test_data)):
    predicted_value = w[0]
    for m in range(len(w)-1):
      predicted_value += w[m+1]*test_data[i][m]

    y_pred.append(predicted_value)
    y_true.append(test_data[i][len(w)-1])
    p=  test_data[i][len(w)-1]  - predicted_value  
    q = test_data[i][len(w)-1]  - mean
    res_sum+=p**2
    tot_sum+=q**2
    
print("r2  is : " , 1-res_sum/tot_sum)
print("\n")

plt.plot(epoch,loss)
plt.xlabel('epoch') 
plt.ylabel('loss')
plt.show()

#calculating RMSE value 
rmse =0 
for i in range(len(y_pred)):
    rmse += pow((y_pred[i] - y_true[i]),2)

rmse/=len(y_pred) 

rmse = math.sqrt(rmse) 
print("RMSE value is : " , rmse)
