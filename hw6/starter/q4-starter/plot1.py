import matplotlib.pyplot as plt
import numpy as np

from models import *
from starter import *


def main():
    #############################################################################
    #######################PLOT PART 1###########################################
    #############################################################################
    np.random.seed(0)

    ns = np.arange(10, 310, 20)
    replicates = 5
    num_methods = 6
    num_sets = 3
    mses = np.zeros((len(ns), replicates, num_methods, num_sets))
    linear = 0

    def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
        return generate_dataset(
            sensor_loc,
            num_sensors=k,
            spatial_dim=d,
            num_data=n,
            original_dist=original_dist,
            noise=noise)

    for s in range(replicates):
        sensor_loc = generate_sensors()
        X_test, Y_test = generate_data(sensor_loc, n=1000)
        X_test2, Y_test2 = generate_data(
            sensor_loc, n=1000, original_dist=False)
        for t, n in enumerate(ns):
            X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
            Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]

            # Linear regression:
            mse0 = linear_regression(X, Y, Xs_test[0], Ys_test[0])
            mse1 = linear_regression(X, Y, Xs_test[1], Ys_test[1])
            mse2 = linear_regression(X, Y, Xs_test[2], Ys_test[2])
            mses[t, s, 0, 0] = mse0
            mses[t, s, 0, 1] = mse1
            mses[t, s, 0, 2] = mse2


            ### Second-order Polynomial regression:
            mse0 = poly_regression_second(X, Y, Xs_test[0], Ys_test[0])
            mse1 = poly_regression_second(X, Y, Xs_test[1], Ys_test[1])
            mse2 = poly_regression_second(X, Y, Xs_test[2], Ys_test[2])
            mses[t, s, 1,0] = mse0
            mses[t, s, 1,1] = mse1
            mses[t, s, 1,2] = mse2

            # ### 3rd-order Polynomial regression:
            # mse0 = poly_regression_cubic(X, Y, Xs_test[0], Ys_test[0])
            # mse1 = poly_regression_cubic(X, Y, Xs_test[1], Ys_test[1])
            # mse2 = poly_regression_cubic(X, Y, Xs_test[2], Ys_test[2])
            # mses[t, s, 2,0] = mse0
            # mses[t, s, 2,1] = mse1
            # mses[t, s, 2,2] = mse2

            #
            ### Neural Network:
            # mse = neural_network(X, Y, Xs_test[0], Ys_test[0])
            # mse = neural_network(X, Y, Xs_test[1], Ys_test[1])
            # mse = neural_network(X, Y, Xs_test[2], Ys_test[2])
            # mses[t, s, 3, 0] = mse
            # mses[t, s, 3, 1] = mse
            # mses[t, s, 3, 2] = mse

            ## Generative model:
            # mse = generative_model(X, Y, Xs_test, Ys_test)
            # mses[t, s, 4] = mse
            # #
            # ### Oracle model:
            # mse = oracle_model(X, Y, Xs_test, Ys_test, sensor_loc)
            # mses[t, s, 5] = mse

            print('{}th Experiment with {} samples done...'.format(s, n))
    print (linear)
    ## Plot MSE for each model.
    plt.figure()
    regressors = [
        'Linear Regression', '2nd-order Polynomial Regression',
        '3rd-order Polynomial Regression', 'Neural Network',
        'Generative Model', 'Oracle Model'
    ]
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 0], axis=1), label=regressors[a])

    plt.title('Error on training data for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('train_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 1], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from the same distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_same_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 2], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from a different distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_different_mse.png')
    plt.show()


if __name__ == '__main__':
    main()
