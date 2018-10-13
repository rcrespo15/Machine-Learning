#!/usr/bin/env python3
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import toolbox
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from numpy.linalg import inv
from numpy import linalg as LA
import math

# choose the data you want to load
# data = np.load('circle.npz')
# data = np.load('heart.npz')
# data = np.load('asymmetric.npz')
data_complete = ['circle.npz','heart.npz','asymmetric.npz']



class HW4_Sol(object):
    def _init_(self):
        self.LAMBDA = 0.001

    def load_data(self,data_file):
        self.LAMBDA = 0.001
        self.data = np.load(data_file)
        self.predictions = np.zeros((3,21056,16))
        self.X = self.data["x"]
        self.y = self.data["y"]
        self.X /= np.max(self.X)  # normalize the data
        self.n, self.d = self.X.shape
        self.values = np.arange(self.n)
        # np.random.shuffle(self.values)
        self.n_train = int(self.n*.8)
        self.n_test = self.n - self.n_train
        self.X_train = self.X[0:self.n_train,:]
        self.y_train = self.y[0:self.n_train]

        self.X_test = self.X[self.n_train+1:self.n,:]
        self.y_test = self.y[self.n_train+1:self.n]

    def return_cost(self, X, y, w):
        J = math.sqrt(np.sum((X.dot(w)-y)**2))
        return (J)

    def return_cost_squared(self,X, y, w):
        J = np.sum((X.dot(w)-y)**2)
        return (J)

    def lstsq(self, A, b, lambda_=0):
        return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ b)

    def heatmap(self, f, clip = 5):
        # example: heatmap(lambda x, y: x * x + y * y)
        # clip: clip the function range to [-clip, clip] to generate a clean plot
        #   set it to zero to disable this function
        self.xx0 = self.xx1 = np.linspace(np.min(self.X), np.max(self.X), 72)
        self.x0, self.x1 = np.meshgrid(self.xx0, self.xx1)
        self.x0, self.x1 = self.x0.ravel(), self.x1.ravel()
        self.z0 = f(self.x0, self.x1)

        if clip:
            self.z0[self.z0 > 5] = 5
            self.z0[self.z0 < -5] = -5

        plt.hexbin(self.x0, self.x1, C=self.z0, gridsize=50, cmap=cm.jet, bins=None)
        plt.colorbar()
        self.cs = plt.contour(
            self.xx0, self.xx1, self.z0.reshape(self.xx0.size, self.xx1.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=cm.jet)
        plt.clabel(self.cs, inline=1, fontsize=10)

        self.pos = self.y[:] == +1.0
        self.neg = self.y[:] == -1.0
        plt.scatter(self.X[self.pos, 0], self.X[self.pos, 1], c='red', marker='+')
        plt.scatter(self.X[self.neg, 0], self.X[self.neg, 1], c='blue', marker='v')
        plt.show()

    def assemble_feature(self, x, D):
        from scipy.special import binom
        xs = []
        for d0 in range(D + 1):
            for d1 in range(D - d0 + 1):
                # non-kernel polynomial feature
                # xs.append((x[:, 0]**d0) * (x[:, 1]**d1))
                # # kernel polynomial feature
                xs.append((x[:, 0]**d0) * (x[:, 1]**d1) * np.sqrt(binom(D, d0) * binom(D - d0, d1)))
        return np.column_stack(xs)

    def sol_2b(self,max_polynomial):
        self.results_2b = np.zeros((max_polynomial,2))
        for i in range(1,max_polynomial,1):
            self.Xd_train = self.assemble_feature(self.X_train, i)
            self.Xd_test = self.assemble_feature(self.X_test, i)
            self.w_ridge = self.lstsq(self.Xd_train, self.y_train, self.LAMBDA )
            self.results_2b[i-1,0] = (self.return_cost_squared(self.Xd_train, self.y_train, self.w_ridge))/self.n_train
            self.results_2b[i-1,1] = (self.return_cost_squared(self.Xd_test, self.y_test, self.w_ridge))/self.n_test
            if i in [2, 4, 6, 8, 10, 12]:
                fname = "result/asym%02d.pdf" % i
                self.heatmap(lambda x, y: self.assemble_feature(np.vstack([x, y]).T, i) @ self.w_ridge, fname)
        print (self.results_2b)

    def sol_2c(self, max_polynomial):
        self.cost_2c = np.zeros((max_polynomial))
        for p in range(1,max_polynomial,1):
            self.K = np.zeros((self.n_train,self.n_train))
            for i in range(self.n_train):
                for j in range(self.n_train):
                    self.K[i,j] = (self.X_train[i].dot(self.X_train[j].T)+1)**p
            self.k = np.zeros([self.n_train])
            self.y_prediction = np.zeros([self.n_train])
            for i in range(self.n_train):
                self.k[:] = (self.X_train[i].T.dot(self.X_train.T)+1)**p
                self.y_prediction[i] = ((self.k.dot(inv(self.K +(self.LAMBDA * np.identity(self.n_train))))).dot(self.y_train))
            self.cost_2c[p-1] = (np.sum((self.y_prediction-self.y_train)**2))/self.n_train
        print (self.cost_2c)
    #result for train

    # [[9.96937480e-01 1.00038828e+00 1.01941027e+00 9.98440975e-01
    #   1.02983468e+00 5.48558321e-01 5.51123551e-01 1.01548699e-01
    #   9.84661852e-02 5.44299017e-02 1.83792153e-02 1.14036173e-02
    #   8.59753780e-03 4.84470035e-03 2.48749814e-03 1.80580103e-03]
    #
    #  [9.54763119e-01 1.90016389e-01 9.06800337e-02 9.12298251e-03
    #   2.98745482e-03 1.61732299e-03 1.05036273e-03 4.26754871e-04
    #   2.02665193e-04 1.38913938e-04 1.14977505e-04 9.71477741e-05
    #   8.45903338e-05 7.55360636e-05 6.84339794e-05 6.22037094e-05]
    #
    #  #results for test
    #
    #  [[9.97087613e-01 9.95536830e-01 9.92698937e-01 9.43011263e-01
    #   9.35538811e-01 5.11241434e-01 5.07592479e-01 8.63891581e-02
    #   8.18092527e-02 4.30864829e-02 1.39664117e-02 8.68474388e-03
    #   6.51656969e-03 3.66509031e-03 1.91170214e-03 1.39993874e-03]
    #  [9.62643034e-01 2.36718213e-01 1.15480764e-01 1.21631329e-02
    #   3.75913956e-03 2.29395081e-03 1.44087706e-03 6.65027637e-04
    #   3.05465895e-04 1.88558125e-04 1.39349161e-04 1.10966040e-04
    #   9.32048470e-05 8.11449373e-05 7.18264506e-05 6.39087388e-05]
    #



    # Question c.b - Implement kernel ridge regression to cicrle, hear and
    # asymetric data sets
    # coef = .001
    # cost_avg = np.zeros((24,2))
    # data = np.load(data_complete[0])
    # X = data["x"]
    # y = data["y"]
    # X /= np.max(X)  # normalize the data
    # n,d = X.shape
    # # for i in range(10):
    # values = np.arange(n)
    # # np.random.shuffle(values)
    # train_count = int(n*.15)
    # X_train = X[0:train_count,:]
    # n_train,d_train = X_train.shape
    # y_train = y[0:train_count]
    #
    # X_test = X[train_count+1:n,:]
    # n_test,d_test = X_test.shape
    # y_test = y[train_count+1:n]
    # for p in range(1,25,1):
    #     K = np.zeros((n_train,n_train))
    #     for i in range(n_train):
    #         for j in range(n_train):
    #             K[i,j] = ((X_train[i].T.dot(X_train[j]))+1)**p
    #     poly = PolynomialFeatures(p)
    #     X_train_augmented = poly.fit_transform(X_train)
    #     X_test_augmented = poly.fit_transform(X_test)
    #     w = ((X_train_augmented.T.dot(inv(K +
    #         (coef * np.identity(n_train))))).dot(y_train))
    #     cost_avg[p-1,0] = (1/n_train)*return_cost_squared(X_train_augmented, y_train, w)
    #     cost_avg[p-1,1] = (1/n_test)*return_cost_squared(X_test_augmented, y_test, w)
    # print (cost_avg)

    # Question 3.d --> Diminishing influence of the prior wtg growing amount of
    #data
    # coef = np.array([.0001,.001,.01])
    # X_size = np.array([10,20,30,40,70,100,400,700,1000,2000,4000,6000,9000,12000,14000,16000])
    # possible_p = np.array([5,6])
    # data = np.load(data_complete[2])
    # X = data["x"]
    # y = data["y"]
    # X /= np.max(X)  # normalize the data
    # n,d = X.shape
    # cost = np.zeros((16,3,2))
    # for z in range(len(X_size)):
    #     for t in range(2):
    #         count = 0
    #         for i in range(5,7,1):
    #             cost_train = 0
    #             cost_test  = 0
    #             for p in range(100):
    #                 #shuffle data
    #                 values = np.arange(n)
    #                 np.random.shuffle(values)
    #                 train_values = values[0:X_size[z]]
    #
    #                 X_train = X[train_values,:]
    #                 y_train = y[train_values]
    #
    #                 X_test = X[int(n*.8):n,:]
    #                 y_test = y[int(n*.8):n]
    #                 #fit polynomial to x data
    #                 poly = PolynomialFeatures(i)
    #                 X_train_augmented = poly.fit_transform(X_train)
    #                 n_train,_ = X_train_augmented.shape
    #                 X_test_augmented = poly.fit_transform(X_test)
    #                 n_test,_ = X_test_augmented.shape
    #                 w = toolbox.train_data_ridge(X_train_augmented, y_train ,coef[t])
    #
    #                 cost_train = (return_cost_squared(X_train_augmented,y_train,w))/n_train
    #                 cost_test  += (return_cost_squared(X_test_augmented,y_test,w))/n_test
    #             cost[z,t,count] = cost_test/100
    #             count += 1
    #     # heatmap(lambda x0, x1: predictions[t,0:n,1].T)
    # print (cost)
    # # for i in range(3):
    # plt.xscale('log')
    # for i in range(3):
    #     for t in range(2):
    #         plt.plot(X_size,cost[:,i,t],label = "lambda -> " + str(coef[i]) + ", p -> " +str(possible_p[t]))
    # plt.title("Prior vs Data Size for data -> " + str(data_complete[i]))
    # plt.ylabel("Squared Average Error")
    # plt.xlabel("Train Data Size")
    # plt.legend()
    # plt.show()


    # Question e - Implement RBF kernel ridge regression to heart
    # coef = .001
    # sigma = np.array([10,3,1,.3,.1,.03])
    # data = np.load(data_complete[1])
    # X = data["x"]
    # y = data["y"]
    # X /= np.max(X)  # normalize the data
    # n,d = X.shape
    # values = np.arange(n)
    # np.random.shuffle(values)
    # train_count = int(n*.8)
    # X_train = X[values[0:train_count],:]
    # n_train,d_train = X_train.shape
    # y_train = y[values[0:train_count]]
    #
    # X_test = X[values[train_count+1:n],:]
    # n_test,d_test = X_test.shape
    # y_test = y[values[train_count+1:n]]
    # cost_avg = np.zeros((len(sigma),2))
    # for p in range(len(sigma)):
    #     K = np.zeros((n_train,n_train))
    #     for i in range(n_train):
    #         for j in range(n_train):
    #             K[i,j] = np.exp(-((LA.norm((X_train[i]-X_train[j]),2))**2)/(2*(sigma[p]**2)))
    #         k = np.zeros([n_train])
    #         y_prediction = np.zeros([n_train])
    #         for i in range(n_train):
    #             k[:] = np.exp(-((LA.norm((X_train[i]-X_train),2))**2)/(2*(sigma[p]**2)))
    #             y_prediction[i] = ((k.dot(inv(K +(coef * np.identity(n_train))))).dot(y_train))
    #         cost_avg[p-1,0] = (np.sum((y_prediction-y_train)**2))/n_train
    # print (cost_avg)
    # plt.plot(sigma,cost_avg[:,0],label = "avg train error ")
    # # plt.plot(sigma,cost_avg[:,1],label = "avg test error ")
    # plt.title("RBF Kernel Avg Square Loss")
    # plt.ylabel("Squared Average Error")
    # plt.xlabel("Sigmas")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    hw4_sol = HW4_Sol()
    hw4_sol.load_data('circle.npz')
    # hw4_sol.sol_2b(16)
    hw4_sol.sol_2c(17)
