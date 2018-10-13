from __future__ import division
import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import os
import math

#Actual code to use once data has been prototyped

# class HW3_Sol(object):
#
#     def __init__(self):
#         pass
#
#     def load_data(self):
#         self.x_train = pickle.load(open('x_train.p','rb'), encoding='latin1')
#         self.y_train = pickle.load(open('y_train.p','rb'), encoding='latin1')
#         self.x_test = pickle.load(open('x_test.p','rb'), encoding='latin1')
#         self.y_test = pickle.load(open('y_test.p','rb'), encoding='latin1')
#
#
# if __name__ == '__main__':
#
#     hw3_sol = HW3_Sol()

     # Your solution goes here

#Part 1 - define some useful functions
def visualize_digit(features, name):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features, interpolation='nearest')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Image from datapoint ' + '_'+str(name) +'.png')
    plt.savefig('Image from datapoint ' + '_'+str(name) +'.png')
    plt.show()

def train_data_ols(X, y):
    w = ((inv(X.T.dot(X))).dot(X.T)).dot(y) #((X.T*X)^-1)*X*y
    return w

def train_data_ridge(X, y, l):
    n,d = X.shape
    I = np.identity(d)
    w = ((inv((X.T.dot(X))+l*I)).dot(X.T)).dot(y) #((X.T*X+l)^-1)*X*y
    return w

def euclidean_distance(X,w,y):
    n,d = X.shape
    t,outputs = y.shape
    distance = np.zeros(outputs)
    for i in range(outputs):
        distance[i] = (1/n)*sum(np.square(X.dot(w[:,i])-y[:,i]))
    return sum(distance)

#Part 2 - import the data
x_train = pickle.load(open('x_train.p','rb'), encoding='latin1')
y_train = pickle.load(open('y_train.p','rb'), encoding='latin1')
x_test  = pickle.load(open('x_test.p','rb'), encoding='latin1')
y_test  = pickle.load(open('y_test.p','rb'), encoding='latin1')

print(y_train.shape,x_test.shape,y_test.shape)
#Problem A - Visualize the 0th, 10th and 20th image
# visulization_array = np.array([0,9,19])
# for i in range(3):
#     visualize_digit(x_train[visulization_array[i]],visulization_array[i])
#     print (y_train[i])

#Problem B
#Compose matrix X_train_bar from x_train. x E R(n x 2700)
n,x,y,z = x_train.shape
X_train = x_train.reshape(n,2700)
#load n examples from y_train and compose the matrix U where U E R(nx3)
U_train = y_train
#Perform OLS to optimize pi
#mute due to error
# pi = train_data_ols(X_train,U_train)

#Problem C
#Perform ridge regression for lambda = [.1,1,10,100,1000]
l = np.array([.1, 1, 10, 100, 1000])


w = np.zeros((2700,3))
euclidean = np.zeros(5)

#
# for i in range(len(l)):
#     for U in range(3):
#         pi = train_data_ridge(X_train, U_train[:,U], l[i])
#         w[:,U] = pi
#     euclidean[i] = euclidean_distance(X_train,w,U_train)
# print ("""The Eucledian distances for X_train, y_train and w and for the lambda
#  values of 0.1,1.0,10,100,1000 are:""")
# print (euclidean)
#
#
# #Problem D
# #Perform ridge regression for lambda = [.1,1,10,100,1000] by normalizing data
# X_train_bar = ((X_train/255)*2) - 1
#
# w_bar = np.zeros((2700,3))
# euclidean_bar = np.zeros(5)
#
#
# for i in range(len(l)):
#     for U in range(3):
#         pi = train_data_ridge(X_train_bar, U_train[:,U], l[i])
#         w_bar[:,U] = pi
#     euclidean_bar[i] = euclidean_distance(X_train_bar,w,U_train)
# print ("""The Eucledian distances for X_train_bar, y_train and w and for the lambda
#  values of 0.1,1.0,10,100,1000 are:""")
# print (euclidean_bar)

#Problem E
#Evaluate both policies on the new validation data
n_test,x_coordinate_test,y_coordinate_test,z_test = x_test.shape
X_test = x_test.reshape(n_test,2700)
U_test = y_test

X_test_bar = ((X_test/255)*2) - 1

w_test = np.zeros((2700,3))
euclidean_test = np.zeros(5)

w_test_bar = np.zeros((2700,3))
euclidean_test_bar = np.zeros(5)



for i in range(len(l)):
    for U in range(3):
        pi = train_data_ridge(X_test, U_test[:,U], l[i])
        w_test[:,U] = pi
    euclidean_test[i] = euclidean_distance(X_test,w_test,U_test)
print ("""The Eucledian distances for X_test, y_test and w_test and for the lambda
 values of 0.1,1.0,10,100,1000 are:""")
print (euclidean_test)


for i in range(len(l)):
    for U in range(3):
        pi = train_data_ridge(X_test_bar, U_test[:,U], l[i])
        w_test_bar[:,U] = pi
    euclidean_test_bar[i] = euclidean_distance(X_test_bar,w,U_test)
print ("""The Eucledian distances for X_test_bar, y_test and w and for the lambda
 values of 0.1,1.0,10,100,1000 are:""")
print (euclidean_test_bar)
