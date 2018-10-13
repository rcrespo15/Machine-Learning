import numpy as np
from numpy.linalg import inv
import math

def return_cost(X, y, w):
    J = math.sqrt(np.sum((X.dot(w)-y)**2))
    return J

def train_data(X, y):
    w = ((inv(X.T.dot(X))).dot(X.T)).dot(y) #((X.T*X)^-1)*X*y
    return w
