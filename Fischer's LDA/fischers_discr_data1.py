#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:52:15 2020

@author: vaibhav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Read dataset 1 and randomly split it into training
# and test set. Test set contains approx. 80% data.
data_1 = pd.read_csv("./../data/a1_data/a1_d1.csv", names=["x", "y", "label"])
mask_1 = np.random.rand(len(data_1)) < 0.8
data_1_train = data_1[mask_1]
data_1_test = data_1[~mask_1]

# Make their indexes as 0,1,2....
# Because of random selection above, their indexes are
# not sequential.
data_1_train = data_1_train.reset_index()
data_1_test = data_1_test.reset_index()


def calc_wts(data):
    """
    Calculate weights required by fishcer's discriminant
    to transform 2-dimensional data to 1-dimensional data.
    """
    
    class0_data = data.loc[data["label"] == 0]
    class1_data = data.loc[data["label"] == 1]
    
    # m0 and m1 are mean of class 0 and class 1 resp.
    m0 = class0_data.loc[:,['x', 'y']].mean()
    m0 = pd.DataFrame(m0)
    m1 = class1_data.loc[:,['x', 'y']].mean()
    m1 = pd.DataFrame(m1)
    
    # Sw is within-class variance. Initialize it with
    # all zeros
    Sw = pd.DataFrame(np.zeros((2,2)), index=['x', 'y'], columns=['x', 'y'])
    
    # Apply the formula 
    # Sw = {n belongs to C0}sum((Xn - m0)(Xn - m0)^T) + {n belongs to C1}sum((Xn - m0)(Xn - m0)^T)
    for index, row in class0_data.iterrows():
        Xn = pd.DataFrame({0: row[['x', 'y']]})
        diff = Xn - m0
        Sw += diff.dot(diff.transpose())
        
    for index, row in class1_data.iterrows():
        Xn = pd.DataFrame({0: row[['x', 'y']]})
        diff = Xn - m1
        Sw += diff.dot(diff.transpose())
        
    Sw = np.array(Sw)
    # Calculate Sw^-1
    Sw_inv = np.linalg.inv(Sw)
    # wts is proportional to Sw_inv * (m2 - m1)
    # Here the constant is taken to be 1
    wts = Sw_inv.dot(np.array(m1 - m0))
    
    return wts

def transform(data, wts):
    """
    Returns data after transforming it from 2-D to 1-D
    for both the classes.
    """
    class0_data = data.loc[data["label"] == 0]
    class1_data = data.loc[data["label"] == 1]
    
    # Empty containers to hold transformed data
    class0_transformed = np.empty((1, class0_data.shape[0]))
    class1_transformed = np.empty((1, class1_data.shape[0]))
    
    # Transform each co-ordinate to 1-D and store it.
    cnt = 0
    for index, row in class0_data.iterrows():
        Xn = pd.DataFrame(row[['x', 'y']])
        class0_transformed[0, cnt] = wts.transpose().dot(Xn).iloc[0, 0]
        cnt += 1
    
    cnt = 0
    for index, row in class1_data.iterrows():
        Xn = pd.DataFrame(row[['x', 'y']])
        class1_transformed[0, cnt] = wts.transpose().dot(Xn).iloc[0, 0]
        cnt += 1

    return (class0_transformed, class1_transformed)
    
def plot_norm_dist(pnts):
    """
    For a given set of points, plot normal distrubution 
    followed by those points.
    """
    # Following line changes the dimension of points
    # TASK: on console try, pnts.shape, pnts[0, :].shape
    pnts = pnts[0, :]
    pnts = np.sort(pnts)
    mean, std = stats.norm.fit(pnts)
    pdf = stats.norm.pdf(pnts, mean, std)
    plt.plot(pnts, pdf, linewidth=2)
    return mean, std
    
def find_gauss_inter(m1,m2,std1,std2):
    """
    Find x-axis intersection point of two guassian distr.
    """
    # a, b, c are the coefficients in the quadratic equations
    # obtained by equating two guassian equations.
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    
    # Calculate root of quadratic equation with coeff a, b, c
    return np.roots([a,b,c])

def plot_pnts_linear(class0_transformed, class1_transformed):
    """
    Plot the transformed points on a line.
    """
    plt.plot(class0_transformed, 
             [0 for pnt in range(len(class0_transformed))], 'g*', markersize=1)
    
    plt.plot(class1_transformed, 
             [0 for pnt in range(len(class0_transformed))], 'ro', markersize=1)

def plot_intersection(m1, std1, m2, std2):
    """
    Plot the points obtained by the intersection of two 
    guassian distr with given means and stddevs.
    """
    roots = find_gauss_inter(m1, m2, std1, std2)
    # Below line can be used, to plot both their intersection points
    # plt.plot(roots[0], stats.norm(m1, std1).pdf(roots[0]), 'ro')
    plt.plot(roots[1], stats.norm(m1, std1).pdf(roots[1]), 'go')
    
def predict(split_pnt, x):
    """
    Predict class of a point given intersection of
    two guassian curves. If point lies to the left return
    class 0 else return class 1.
    """
    if x <= split_pnt:
        return 0
    else:
        return 1
    
def calc_accuracy(data, wts, split_pnt):
    """
    Given a dataset and weights, transforms the points to
    1 dimensional space, and based on intersection point,
    returns the accuracy, precision, recall and f-score
    of given weights and split_pnt.
    """
    correct_predictions = 0
    pos_examples = 0
    pos_predictions = 0
    true_pos = 0
    
    for index, point in data.iterrows():
        # Calculate number of positive examples in the dataset.
        if int(point['label']) == 0:
            pos_examples += 1
            
        Xn = pd.DataFrame({0: point[['x', 'y']]})
        # Transform points to 1-D space.
        transformed = wts.transpose().dot(Xn)[0][0]
        
        prediction = predict(split_pnt, transformed)
        
        # If predicted output is same as actual output
        # then increase correct predictions.
        if prediction == int(point['label']):
            correct_predictions += 1
            
        # Calculate number of positive predictions.
        if prediction == 0:
            pos_predictions += 1
        
        # Calculate True positives.
        if prediction == 0 and int(point['label']) == 0:
            true_pos += 1    
            
    accuracy = correct_predictions / len(data)
    precision = true_pos / pos_predictions
    recall = true_pos / pos_examples
    f_score = (2 * precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f_score

def main():
    #calculations for data set 1
    wts = calc_wts(data_1_train)
    wts = pd.DataFrame(wts, index=['x', 'y'])
    (c1_pnts, c2_pnts) = transform(data_1, wts)
    
    (mean1, std1) = plot_norm_dist(c1_pnts)
    (mean2, std2) = plot_norm_dist(c2_pnts)
    
    plot_intersection(mean1, std1, mean2, std2)
    plot_pnts_linear(c1_pnts, c2_pnts)
    plt.savefig("./../graphs/fischers_discr_data_1.png")
    plt.show()
    
    intersection = find_gauss_inter(mean1, mean2, std1, std2)
    acc = calc_accuracy(data_1_test, wts, intersection[1])
    print("Data 1 - Weights", wts)
    print("Data 1 - Intersection point", intersection)
    print("Data 1 - Accuracy, precision, recall, f-score : ", acc)
    return 0
    
if __name__ == "__main__":
    main()   
  