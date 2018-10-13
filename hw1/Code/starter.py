import numpy as np
import matplotlib.pyplot as plt
import os
import linear_regression as lr
from numpy.linalg import inv
import math
from __future__ import division


d = os.getcwd()
# Load the training dataset
train_features = np.load(os.path.join(d,"train_features.npy"))
train_labels = np.load("train_labels.npy").astype("int8")
n_train = train_labels.shape[0]


def visualize_digit(features, label, digit, name):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))
    plt.savefig('Digit '+ str(digit) + '_'+str(name) +'.png')
    plt.show()

# Visualize a digit
# visualize_digit(train_features[0,:], train_labels[0])

# TODO: Plot three images with label 0 and three images with label 1
value_zeros = 0
value_ones  = 1
A = np.where(train_labels==value_zeros)[0]
B = np.where(train_labels==value_ones)[0]

values = [0,0,0,1,1,1]
for i in range(0,3,1):
    visualize_digit(train_features[A[i],:], train_labels[A[i]],values[i],i)
    visualize_digit(train_features[B[i],:], train_labels[B[i]],values[i+3],i)

# Linear regression

# TODO: Solve the linear regression problem, regressing
# X = train_features against y = 2 * train_labels - 1

def train_data(X, y):
    w = ((inv(X.T.dot(X))).dot(X.T)).dot(y) #((X.T*X)^-1)*X*y
    return w

X = train_features
y = 2*train_labels - 1
y_unmodified = train_labels
w = train_data(X,y)
# TODO: Report the residual error and the weight vector

def return_cost(X, y, w):
    J = math.sqrt(np.sum((X.dot(w)-y)**2))
    return J
w = train_data(X,y)
J = return_cost(X,y,w)
# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")
Xprime = test_features
y_prime = test_labels
n_test = test_labels.shape[0]

# TODO: Implement the classification rule and evaluate it
# on the training and test set
def predict(X_input,y_input,w_input,limit_input):
    y_prediction = X_input.dot(w_input)
    y_prediction[y_prediction >  limit_input] = 1
    y_prediction[y_prediction <= limit_input] = 0
    num_correct = np.sum(y_input == y_prediction)
    percentage_correct = (num_correct/len(y_input))*100
    return num_correct,percentage_correct


y_train_correct = predict(X,y_unmodified,w,0)
print ("The train set predicted " + str(y_train_correct[1]) + "% of the values")

test_correct = predict(Xprime,y_prime,w,0)
print ("The algorithm predicted " + str(test_correct[1]) + "% of the values")

# TODO: Try regressing against a vector with 0 for class 0
# and 1 for class 1
#redefine parameters
X = train_features
y = train_labels
Xprime = test_features
yprime = test_labels

#Results for train data
w = train_data(X,y)
J = return_cost(X,y,w)
y_train_prediction = predict(X,y,w,.5)
print ("The algorithm predicted " + str(y_train_prediction [1]) + "% of the values in the training set")

#Results for test data
yprime_test_prediction = predict(Xprime,yprime,w,.5)
print ("The algorithm predicted " + str(yprime_test_prediction[1]) + "% of the values in the test set")

# TODO: Form a new feature matrix with a column of ones added
# and do both regressions with that matrix
#Part 1 --> Add bias column to X vector
X_bias = np.insert(X, 0, 1, axis=1)
Xprime_bias = np.insert(Xprime, 0, 1, axis=1)
Xprime_bias.shape
#Part 2 --> Evaluate performance using y = 2*y-1
y = train_labels
y_modified = 2*train_labels - 1
yprime = test_labels

w_bias = train_data(X_bias,y_modified)
ybias_prediction_train = predict(X_bias,y,w_bias,0)
print ("The algorithm predicted " + str(ybias_prediction_train[1]) + "% of the values in the training set")
ybias_prediction_test = predict(Xprime_bias,yprime,w_bias,0)
print ("The algorithm predicted " + str(ybias_prediction_test[1]) + "% of the values in the test set")

#Part 3 --> Evaluate performance using y = y
y = train_labels
y_modified = train_labels
yprime = test_labels

w_bias = train_data(X_bias,y_modified)
ybias_prediction_train_1 = predict(X_bias,y,w_bias,.5)
print ("The train set predicted " + str(ybias_prediction_train_1[1]) + "% of the values")
ybias_prediction_test_1 = predict(Xprime_bias,yprime,w_bias,.5)
print ("The train set predicted " + str(ybias_prediction_test_1[1]) + "% of the values")

# Logistic Regression

# You can also compare against how well logistic regression is doing.
# We will learn more about logistic regression later in the course.

import sklearn.linear_model

lr = sklearn.linear_model.LogisticRegression()
lr.fit(X, train_labels)

test_error_lr = 1.0 * sum(lr.predict(test_features) != test_labels) / n_test
