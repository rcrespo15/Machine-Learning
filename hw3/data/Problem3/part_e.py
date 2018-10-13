from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv
import math
from toolbox import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def return_cost(X, y, w):
    n,d = X.shape
    J = math.sqrt(np.sum((X.dot(w)-y)**2))
    average_error = J/n
    return (average_error)

def train_data_ridge(X, y,coef):
    n,d = X.shape
    w = ((inv(X.T.dot(X) + (coef * np.identity(d)))).dot(X.T)).dot(y)
    return w

polynomial_degree = np.array([1,2,3,4,5,6,7,9,10])
polynomial_degree2 = np.array([1,2,3,4,5,6,7,9])
data_samples = np.array([10,30,50,80,100,200,500,1000])

cost_function = np.zeros((10,8))

for t in range(len(data_samples)):
    n = data_samples[t]
    i = 2
    w1 = np.ones((2)).T
    sigma = np.ones((n,i))
    sigma[:,1] = np.random.uniform(-1,1,n)
    error =  np.random.normal(0,.5,n)
    y = sigma.dot(w1) + error
    y_true = sigma.dot(w1)

    for i in range(9):
        features = polynomial_degree[i-1]
        poly = PolynomialFeatures(features)
        X = poly.fit_transform(sigma)
        w = train_data_ridge(X,y,.1)
        cost_function[t-1,i-1] = return_cost(X, y_true, w)
print (cost_function)
plt.plot(polynomial_degree2,cost_function[2,:],label = "Rndm",c="b")
# plt.plot(range(7),error[t][:,1],label = "PCA",c="r")
# plt.title("Accuracy Dataset" + str(t+1))
# plt.xlabel("k Value")
# plt.ylabel("Accuracy")
# plt.legend()
plt.show()
plt.plot(data_samples,cost_function[:8,4],label = "Rndm",c="b")
plt.show()
