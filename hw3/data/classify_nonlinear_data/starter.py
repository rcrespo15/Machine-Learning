import numpy as np
import matplotlib.pyplot as plt
from toolbox import *

Xtrain = np.load("Xtrain.npy")
ytrain = np.load("ytrain.npy")

def visualize_dataset(X, y):
    plt.scatter(X[y < 0.0, 0], X[y < 0.0, 1])
    plt.scatter(X[y > 0.0, 0], X[y > 0.0, 1])
    plt.show()

# visualize the dataset:
visualize_dataset(Xtrain, ytrain)

# TODO: solve the linear regression on the training data

Xtest = np.load("Xtest.npy")
ytest = np.load("ytest.npy")

#Part b.
#b.1 --> Use linear regression to train w with test data.
w1 = train_data_ols(Xtest, ytest)
#b.2 --> Use trained w to predict accuracy on test data
# TODO: report the classification accuracy on the test set
number_correct, percentage_correct = predict(Xtest,ytest,w1,0)

print (percentage_correct)

#Part c.
#c.1 --> Augment the matrix adding polynomial features 1,x2,xy,y2.
# TODO: Create a matrix Phi_train with polynomial features from the training data
# and solve the linear regression on the training data
n,d = Xtrain.shape
n_test,d_test = Xtest.shape
Phi_train = np.ones((n,1))
Phi_train = np.insert(Phi_train, 1, (Xtrain[:,0]), axis =1)
Phi_train = np.insert(Phi_train, 2, (Xtrain[:,1]), axis =1)
Phi_train = np.insert(Phi_train, 3, (Xtrain[:,0]**2), axis =1)
Phi_train = np.insert(Phi_train, 4, (Xtrain[:,0]*Xtrain[:,1]), axis =1)
Phi_train = np.insert(Phi_train, 5, (Xtrain[:,1]**2), axis =1)


#fit Polynomial
w2 = train_data_ols(Phi_train,ytrain)
number_correct_train, percentage_correct_train = predict(Phi_train,ytrain,w2,0)
print ("The weights vector to fit the polynomial is ")
print (w2)
# TODO: Create a matrix Phi_test with polynomial features from the test data
# and report the classification accuracy on the test set

Phi_test = np.ones((n_test,1))
Phi_test = np.insert(Phi_test, 1, (Xtest[:,0]), axis =1)
Phi_test = np.insert(Phi_test, 2, (Xtest[:,1]), axis =1)
Phi_test = np.insert(Phi_test, 3, (Xtest[:,0]**2), axis =1)
Phi_test = np.insert(Phi_test, 4, (Xtest[:,0]*Xtest[:,1]), axis =1)
Phi_test = np.insert(Phi_test, 5, (Xtest[:,1]**2), axis =1)

number_correct2, percentage_correct2 = predict(Phi_test,ytest,w2,0)


print (percentage_correct2)

#Part d.
#d --> Prove that the classification rule has the form ax2 _ ay2 <=B
Phi_train_c = np.ones((n,1))
Phi_train_c = np.insert(Phi_train_c, 1, (Xtrain[:,0]**2), axis =1)
Phi_train_c = np.insert(Phi_train_c, 2, (Xtrain[:,0]**2), axis =1)

Phi_test_c = np.ones((n_test,1))
Phi_test_c = np.insert(Phi_test_c, 1, (Xtest[:,1]**2), axis =1)
Phi_test_c = np.insert(Phi_test_c, 2, (Xtest[:,1]**2), axis =1)

w3 = train_data_ols(Phi_train_c,ytrain)
number_correct3, percentage_correct3 = predict(Phi_train_c,ytrain,w3,0)
print (percentage_correct3)
