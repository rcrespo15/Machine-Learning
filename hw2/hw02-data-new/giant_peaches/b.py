#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from numpy.linalg import inv


# There is numpy.linalg.lstsq, whicn you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)

def main():
    data = spio.loadmat('1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T

    n = 20  # max degree
    err = np.zeros(n)

    # fill in err
    # YOUR CODE HERE
    #redefine x_train to include a 1 vector

    D = np.arange(1,21,1)
    d = len(D)

    for i in range(d):
        X_train = np.ones((n,1))
        for t in range(1,D[i],1):
            X_train = np.insert(X_train, t, (x_train ** t), axis =1)
        #fit Polynomial
        w = lstsq(X_train,y_train)
        err[i] = (1/20)*sum((X_train.dot(w)-y_train)**2)

    plt.plot(err)
    plt.ylim([0, 6])
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Training Error')
    plt.show()


if __name__ == "__main__":
    main()
