import numpy as np
import matplotlib.pyplot as plt

sample_size = [5,25,125,625]
plt.figure(figsize=[12, 10])
w_true = .5 #true value of the model
w_test = np.linspace(.3,.7,num=11) #range of values of w to test and visualiza
# w_test = np.array([.45])
N = len(w_test)


#Function that returs the likelihood of a uniform function

#uniform_likelihood returns the likelihood of a function given that the function
#has a uniform distribution for all the values of x in the acceptable range
#X = matrix of the input parameters
#w = hyperparameter, scalar
#w_test = hyper parameter to be tested. The program will evaluate the likelihood
#         of y given this w_test parameters. Vector.
# upper_limit and lower_limit is the range on which the function operates.
def uniform_likelihood(X,w,w_test,upper_limit,lower_limit):
    n = len(X)
    Y_true  = X*w
    U = np.random.uniform(-.5,.5,n)
    likelihood = np.ones(len(w_test))

    for w in range(len(w_test)):
        probability = np.ones(n)
        Y_error = X*w_test[w] + U #array with all the values of Y_error

        for i in range(n):
            gap = Y_true[i] - Y_error[i]
            if gap <= upper_limit and gap >= lower_limit:
                probability[i] = 1
            else:
                probability[i] = 0
        likelihood[w] = np.prod(probability)
    return likelihood

for k in range(len(sample_size)):
    n = sample_size[k]

    # generate data
    # np.linspace, np.random.normal and np.random.uniform might be useful functions
    X = np.linspace(1, 2, num=n)
    # X = np.random.normal(, 10, num=n)
    likelihood = uniform_likelihood(X,w_true,w_test,.5,-.5)

    # likelihood = sum(likelihood) # normalize the likelihood

    plt.figure()
    # plotting likelihood for different n
    plt.plot(w_test, likelihood)
    plt.xlabel('w', fontsize=10)
    plt.title(['n=' + str(n)], fontsize=14)

plt.show()
