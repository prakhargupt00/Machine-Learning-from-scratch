"""
Created on Fri Oct 18 21:35:00 2019
"""

#GRADIENT DESCENT ALGORITHM FOR LINEAR REGRESSION 
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

# data standardisation
data[0:,1]= (data[0:,1] -np.mean(data[0:,1], dtype=np.float64)) / np.std(data[0:,1], dtype=np.float64)
data[0:,2]= (data[0:,2] -np.mean(data[0:,2], dtype=np.float64)) / np.std(data[0:,2], dtype=np.float64)
data[0:,3]= (data[0:,3] -np.mean(data[0:,3], dtype=np.float64)) / np.std(data[0:,3], dtype=np.float64)

train_data = data[0:304411,1:4]  #ignoring first column
test_data  = data[304411: ,1:4]  # 0 to 304411 is 70% of data
 
#initialise weights
w= np.zeros(3)

#learning rate
alpha = 2e-6

# no of iteration 
k = 10000

#cost
J=0 

loss = []    
epoch = []
#calculating the error function using gradient descent 
for i in range(k):   #num_of_iterations
    h=0
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
    h=0.5*h
    loss.append(h)
    epoch.append(i)
    J=h
    #updating wieghts
    w[0]=w[0]-alpha*(c0)
    w[1]=w[1]-alpha*(c1)
    w[2]=w[2]-alpha*(c2)
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
    
print("r2  is : " , 1-res_sum/tot_sum)
print("\n")

plt.plot(epoch,loss)
plt.xlabel('epoch') 
plt.ylabel('loss')
#plt.savefig("parta.jpg")
plt.show()

#calculating RMSE value 
rmse =0 
for i in range(len(y_pred)):
    rmse += pow((y_pred[i] - y_true[i]),2)

rmse/=len(y_pred) 

rmse = math.sqrt(rmse) 
print("RMSE value is : " , rmse)


