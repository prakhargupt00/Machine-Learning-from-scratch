# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:47:41 2020

@author: Prakhar
"""

#NEURAL NETWORK
import numpy as np 
import pandas as pd 
import math 
import random 
import matplotlib.pyplot as plt
from tqdm import tqdm #for visual representation of iterations 

#sigmoid function
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#RELU activation fuction
def relu(x):
    return np.maximum(x,0)

def relu_der(x):
    return np.greater(x,0).astype(int)

#standardise function 
def standardise(X):
    return (X-np.mean(X)) / np.std(X)

#feed forward function
def feedForward(x,w1,w2,b1,b2):
    hidden = relu(np.dot(x.reshape(1,10), w1) + b1)
    output = sigmoid(np.dot(hidden, w2) + b2)
    
    return hidden,output

#backpropagation function
def backpropagation(x,w1,w2,output,hidden,y,eta,b1,b2):
        # application of the chain rule to find derivative of the loss function 
        # with respect to weights2 and weights1
    
        d_w2 = np.dot(hidden.T.reshape(-1,1), (2*(y - output) * sigmoid_der(output)))
        d_w1 = np.dot(x.T.reshape(-1,1), (np.dot(2*(y - output) * sigmoid_der(output), w2.T) * relu_der(hidden)).reshape((1,5)))        
        
        #derivates for b 
        d_b1 = np.dot(np.ones((1,1)), (np.dot(2*(y - output) * sigmoid_der(output), w2.T) * relu_der(hidden))) 
        d_b2 = np.dot(np.ones((1,1)), (2*(y - output) * sigmoid_der(output))) 
        
        # update the weights with the derivative (slope) of the loss function
        w1 = w1 - eta * d_w1
        w2 = w2 - eta * d_w2
        b2 = b2 - eta * d_b2
        b1 = b1 - eta * d_b1
        return w1,w2,b1,b2

if __name__ == "__main__":
    
    filename = "housepricedata.csv"
    dataset = pd.read_csv(filename)

     #converting the data into numpy array and splitting into test and training data 
    data = dataset.values
    data = data.astype('float64')
    np.random.shuffle(data)
    
    #standardisation 
    for i in range(len(data[0,:])-1):
            data[:,i] = standardise(data[:,i])

    #train_test_split 80-20 
    train_data = data[:1168,:11]
    test_data = data[1168:,:11]
    
    x_train = train_data[:, :10]  
    x_test  = test_data[:, :10]
    y_train = train_data[:, 10]
    y_test  = test_data[:, 10]
    
    #learning rate
    eta = 0.05
    
    #interations
    itr = 1000

    # 2 LAYER NEURAL NETWORK WITH 5 HIDDEN NEURONS(following Bishop Pattern)
    #weight initialisation
    #weight between  input and hidden
    w1 = np.random.rand(x_train.shape[1],5)
    #weight between hidden and output
    w2 = np.random.rand(5,1)
    
    #initialising bias for hidden and output layer 
    b1 = np.zeros((1,5))
    b2 = np.zeros((1,1))

    #Gradient descent
    for i in tqdm(range(itr)):
        error = 0 
        for j in range(len(x_train)):
            hidden, output = feedForward(x_train[j],w1,w2,b1,b2)
            w1,w2,b1,b2 = backpropagation(x_train[j],w1,w2,output,hidden,y_train[j],eta,b1,b2)
            error += (output-y_train[j])**2
        error /= 2
        
        print(error)
        
    
    
    #predicting for test set
    x_test = np.hstack([np.ones((x_test.shape[0], 1)),x_test])
    prediction = []
    
    #calculating accuracy
    y_test =y_test.reshape(-1, 1)
    #correct = prediction == y_test[:,0]
    #accuracy = (np.sum(correct) / len(test_data))*100
    #print ('Logistic regression Accuracy %: ', accuracy)
    
    
    
    
    
