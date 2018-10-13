#contains useful functions
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv
import math

 # np.linspace, np.random.normal and np.random.uniform

#Train data using OLS
#Import -->
# X = matrix ∈ R nxd. n = samples, d features
# y = matrix ∈ R nxs. n = samples, s = output classes
#Output -->
# w = matrix ∈ R nxs. n = number of samples, s number of weights that classify
#     the data into the outputs 1,..,s
def train_data_ols(X, y):
    w = ((inv(X.T.dot(X)+.1)).dot(X.T)).dot(y) #((X.T*X)^-1)*X*y
    return w

#Predict data using w as a weight vector
#This function is written to output the accuray of your predictions y_prediction
#as compared to the measured y
#Import -->
# X = matrix ∈ R nxd. n = samples, d featsures
# y = matrix ∈ R n. n = samples. This formulation only works for two possible
#     outputs separated by a limit.
# w = matrix ∈ R nxs. n = number of samples, s number of weights that classify
#     the data into the outputs 1,..,s. This is also considered the model to be
#     tested.
# limit = scalar. Value that separates the class.
#Output -->
# number of correct predictions
# percentage of correct predictions
def predict(X,y,w,limit):
    y_prediction = X.dot(w)
    y_prediction[y_prediction >  limit] = 1
    y_prediction[y_prediction <= limit] = -1
    num_correct = np.sum(y == y_prediction)
    percentage_correct = (num_correct/len(y))*100
    return num_correct,percentage_correct

def return_cost(X, y, w):
    J = math.sqrt(np.sum((X.dot(w)-y)**2))
    return (J)

def train_data_ridge(X, y,coef):
    n,d = X.shape
    w = ((inv(X.T.dot(X) + (coef * np.identity(d)))).dot(X.T)).dot(y)
    return w
