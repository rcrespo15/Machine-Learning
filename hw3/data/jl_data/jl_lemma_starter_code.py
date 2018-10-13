import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn.linear_model
from sklearn.model_selection import train_test_split


######## PROJECTION FUNCTIONS ##########

## Random Projections ##
def random_matrix(d, k):
    '''
    d = original dimension
    k = projected dimension
    '''
    return 1./np.sqrt(k)*np.random.normal(0, 1, (d, k))

def random_proj(X, k):
    _, d= X.shape
    return X.dot(random_matrix(d, k))

## PCA and projections ##
def my_pca(X, k):
    '''
    compute PCA components
    X = data matrix (each row as a sample)
    k = #principal components
    '''
    n, d = X.shape
    assert(d>=k)
    _, _, Vh = np.linalg.svd(X)
    V = Vh.T
    return V[:, :k]

def pca_proj(X, k):
    '''
    compute projection of matrix X
    along its first k principal components
    '''
    P = my_pca(X, k)
    # P = P.dot(P.T)
    return X.dot(P)


######### LINEAR MODEL FITTING ############

def rand_proj_accuracy_split(X, y, k):
    '''
    Fitting a k dimensional feature set
    obtained from random projection
    of X, versus y for binary classification
    for y in {-1, 1}
    '''

    # test train split
    _, d = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # random projection
    J = np.random.normal(0., 1., (d, k))
    rand_proj_X = X_train.dot(J)

    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(rand_proj_X, y_train)

    # predict y
    y_pred=line.predict(X_test.dot(J))

    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)

def pca_proj_accuracy(X, y, k):
    ''' Fitting a k dimensional feature
    set obtained from PCA projection of X,
    versus y for binary classification for y in
    {-1, 1}
    '''

    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pca projection
    P = my_pca(X_train, k)
    P = P.dot(P.T)
    pca_proj_X = X_train.dot(P)

    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(pca_proj_X, y_train)

     # predict y
    y_pred=line.predict(X_test.dot(P))


    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)


######## LOADING THE DATASETS #########

# to load the data:

data_1 = np.load('data1.npz')
X_1 = data_1['X']
y_1 = data_1['y']
n_1, d_1 = X_1.shape

data_2 = np.load('data2.npz')
X_2 = data_2['X']
y_2 = data_2['y']
n_2, d_2 = X_2.shape

data_3 = np.load('data3.npz')
X_3 = data_3['X']
y_3 = data_3['y']
n_3, d_3 = X_3.shape

X = np.array([X_1,X_2,X_3])
y = np.array([y_1,y_2,y_3])
n = np.array([n_1,n_2,n_3])
d = np.array([d_1,d_2,d_3])

n_trials = 10  # to average for accuracies over random projections

######### YOUR CODE GOES HERE ##########

# Using PCA and Random Projection for:
# Visualizing the datasets
#
#
# Step 1. Top-2 PCA
# obtain positions of 1s and -1s

for i in range(3):
    positive_y = np.where(y[i]==1)[0]
    negative_y = np.where(y[i]==-1)[0]

#Visualize the features of PCA
    pca = my_pca(X[i],2)
    features_pca ,_ = pca.shape
    random = random_matrix(7, 2)
    features_random ,_ = random.shape
    plt.scatter(np.array(range(0,features_pca,1)),pca[:,0],
                s=4,marker = "o",c="b",label = "Feature 1 - PCA")
    plt.scatter(np.array(range(0,features_pca,1)),pca[:,1],
                s=4,marker = "o",c="r", label = "Feature 2 - PCA")
    plt.scatter(np.array(range(0,features_random,1)),random[:,0],
                s=4,marker = "v",c="b", label = "Feature 1 - Random Proj")
    plt.scatter(np.array(range(0,features_random,1)),random[:,1],
                s=4,marker = "v",c="r", label = "Feature 2 - Random Proj")
    plt.xlabel("Feature")
    plt.ylabel("Feature Value")
    plt.title("Top 2 Features PCA and 2-D Random Projection data" + str(i+1))
    plt.legend()
    plt.show()

    # Project and visualize using PCA features into 2D
    pca_projection = pca_proj(X[i], 2)
    random_projection = random_proj(X[i], 2)
    plt.scatter(pca_projection[positive_y,0],pca_projection[positive_y,1],
                s=4,marker = "o",label = "PCA - positive",c="b")
    plt.scatter(pca_projection[negative_y,0],pca_projection[negative_y,1],
                s=4,marker = "o",label = "PCA - negative",c="r")
    plt.title("Top 2 Features PCA Projection data" + str(i+1))
    plt.legend()
    plt.show()
    plt.scatter(random_projection[positive_y,0],random_projection[positive_y,1],
                s=4,marker = "v",label = "Rndm - positive",c="b")
    plt.scatter(random_projection[negative_y,0],random_projection[negative_y,1],
                s=4,marker = "v",label = "Rndm - negative",c="r")
    # plt.scatter(np.array(range(0,features_pca,1)),pca[:,1],s=12,marker = "o",c="r")
    plt.title("2-D Random Projection data" + str(i+1))
    plt.legend()
    plt.show()

# Computing the accuracies over different datasets.
error = np.zeros((3,d[0],2)) #first column is random error, second = PCA error
for t in range(len(X)):
    for i in range(d[0]):
        error_inner_rand = 0
        error_inner_pca = 0
        for s in range(10):
            error_inner_rand += rand_proj_accuracy_split(X[t], y[t], i+1)
            error_inner_pca += pca_proj_accuracy(X[t], y[t], i+1)
        error[t,i,0] = error_inner_rand/10
        error[t,i,1] = error_inner_pca/10

for t in range (len(error)):
    plt.plot(range(7),error[t][:,0],label = "Rndm",c="b")
    plt.plot(range(7),error[t][:,1],label = "PCA",c="r")
    plt.title("Accuracy Dataset" + str(t+1))
    plt.xlabel("k Value")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
# Don't forget to average the accuracy for multiple
# random projections to get a smooth curve.





# And computing the SVD of the feature matrix
sigma = np.zeros((7,3))
for i in range(3):
    _, sigma[:,i], _ = np.linalg.svd(X[i])


plt.plot(range(7),sigma[:,0],label = "Dataset1",c="b")
plt.plot(range(7),sigma[:,1],label = "Dataset2",c="r")
plt.plot(range(7),sigma[:,2],label = "Dataset3",c="orange")
plt.title("Singular values Dataset")
plt.ylabel("Sigma")
plt.legend()
plt.show()
######## YOU CAN PLOT THE RESULTS HERE ########

# plt.plot, plt.scatter would be useful for plotting
