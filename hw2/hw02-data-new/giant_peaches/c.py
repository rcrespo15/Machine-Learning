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
    y_fresh = np.array(data['y_fresh']).T

    n = 20  # max degree
    err_train = np.zeros(n)
    err_fresh = np.zeros(n)

    # fill in err_fresh and err_train
    # YOUR CODE HERE

    D = np.arange(1,21,1)
    d = len(D)

    for i in range(d):
        X_train = np.ones((n,1))
        for t in range(1,D[i],1):
            X_train = np.insert(X_train, t, (x_train ** t), axis =1)
        #fit Polynomial
        w = lstsq(X_train,y_train)
        err_train[i] = (1/20)*sum((X_train.dot(w)-y_train)**2)
        err_fresh[i] = (1/20)*sum((X_train.dot(w)-y_fresh)**2)

    plt.figure()
    plt.ylim([0, 6])
    plt.plot(err_train, label='train')
    plt.plot(err_fresh, label='fresh')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
