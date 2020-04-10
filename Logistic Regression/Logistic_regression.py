"""
Created on Mon Apr 06 20:07:00 2020

@author: Prakhar
"""

#LOGISTIC REGRESSION 
import numpy as np 
import pandas as pd 
import math 
import random 
import matplotlib.pyplot as plt
from tqdm import tqdm #for visual representation of iterations 

#sigmoid function
def sigmoid(x):
    return (1/(1+math.exp(-x)))

#standardise
def standardise(X):
    return (X-np.mean(X, dtype=np.float64)) / np.std(X,dtype = np.float64)
    

if __name__ == "__main__":
    #importing dataset and adding header labels 
    filename = "data_banknote_authentication.csv"
    dataset = pd.read_csv(filename, names = ["feature1", "feature2", "feature3","feature4","class"])
    
    #converting the data into numpy array and splitting into test and training data 
    data = dataset.values
    data = data.astype('float64')
    np.random.shuffle(data)
    
    #standardisation 
    data[:,0] =  standardise(data[:,0])  
    data[:,1] =  standardise(data[:,1]) 
    data[:,2] =  standardise(data[:,2])
    data[:,3] =  standardise(data[:,3])  
    
    #train_test_split 80-20 
    train_data = data[:1098,:5]
    test_data = data[1098:,:5]
    
    #learning rate
    eta = 0.1
    
    #interations
    itr = 10000
    
    #weight  vector 
    w = np.zeros(5)
    
    loss = []    
    epoch = []
    
    #gradient descent approach 
    for i in tqdm(range(itr)):
        #cost
        cost = 0
        #intermediate variables used for weight update
        e = np.zeros(5)
        
        for j in range(len(train_data)):
            h_x = w[0] +  w[1]*train_data[j][0] + w[2]*train_data[j][1] + w[3]*train_data[j][2] + w[4]*train_data[j][3]
            h_x = sigmoid(h_x) 
            cost += train_data[j,-1]*math.log(h_x) + (1-train_data[j,-1])*math.log(1-h_x)    
            
            e[0] +=  (h_x - train_data[j,-1])
            for k in range(len(e)-1):
                e[k+1] +=  (h_x - train_data[j,-1])*train_data[j][k]
          
                    
        #cost after each iteration 
        cost = -1 * (cost/len(train_data))

        loss.append(cost)
        epoch.append(i)
        print(cost)
            
        for k in range(0, len(e)):
            e[k] = e[k]/len(train_data)
        
        #update weights            
        for j in range(0,len(w)):
            w[j] = w[j] - eta*e[j]
    
    
    #predicting for test set
    x_test = test_data[:,:-1]
    x_test = np.hstack([np.ones((x_test.shape[0], 1)),x_test])
    prediction = [1.0/(1 + np.exp(-np.dot(x_test_n, w))) > 0.5 for x_test_n in x_test]
    
    #calculating accuracy
    y_test = data[1098:,-1].reshape(-1, 1)
    correct = prediction == y_test[:,0]
    accuracy = (np.sum(correct) / len(test_data))*100
    print ('Logistic regression Accuracy %: ', accuracy)
    

    


