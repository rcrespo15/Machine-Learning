import tensorflow as tf
import numpy as np
import scipy.spatial
from starter import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


#####################################################################
## Models used for predictions.
#####################################################################

def compute_update(single_obj_loc, sensor_loc, single_distance):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:
    single_obj_loc: 1 * d numpy array.
    Location of the single object.

    sensor_loc: k * d numpy array.
    Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: d-dimensional numpy array.

    """
    loc_difference = single_obj_loc - sensor_loc  # k * d.
    phi = np.linalg.norm(loc_difference, axis=1)  # k.
    grad = loc_difference / np.expand_dims(phi, 1)  # k * 2.
    update = np.linalg.solve(grad.T.dot(grad), grad.T.dot(single_distance - phi))

    return update


def get_object_location(sensor_loc, single_distance, num_iters=20, num_repeats=10):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:

    sensor_loc: k * d numpy array. Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    obj_loc: 1 * d numpy array. The mle for the location of the object.

    """
    obj_locs = np.zeros((num_repeats, 1, 2))
    distances = np.zeros(num_repeats)
    for i in range(num_repeats):
        obj_loc = np.random.randn(1, 2) * 100
        for t in range(num_iters):
            obj_loc += compute_update(obj_loc, sensor_loc, single_distance)

        distances[i] = np.sum((single_distance - np.linalg.norm(obj_loc - sensor_loc, axis=1))**2)
        obj_locs[i] = obj_loc

    obj_loc = obj_locs[np.argmin(distances)]

    return obj_loc[0]


def generative_model(X, Y, Xs_test, Ys_test):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    initial_sensor_loc = np.random.randn(7, 2) * 100
    estimated_sensor_loc = find_mle_by_grad_descent_part_e(
        initial_sensor_loc, Y, X, lr=0.001, num_iters=1000)

    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array(
            [get_object_location(estimated_sensor_loc, X_test_single) for X_test_single in X_test])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def oracle_model(X, Y, Xs_test, Ys_test, sensor_loc):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    sensor_loc: location of the sensors.
    Output:
    mse: Mean square error on test data.
    """
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array([
            get_object_location(sensor_loc, X_test_single)
            for X_test_single in X_test
        ])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def linear_regression(X, Y, Xs_test, Ys_test):
    """
    This function performs linear regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """

    ## YOUR CODE HERE
    #################
    w = ((((X.T.dot(X))**-1).dot(X.T)).dot(Y))
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = w.T.dot(Xs_test[i])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2)))
        mses.append(mse)
    return np.average(mses)


def poly_regression_second(X, Y, Xs_test, Ys_test):
    """
    This function performs second order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    X_poly = assemble_feature(X, 2)
    X_poly_test = assemble_feature(Xs_test, 2)

    w = ((((X.T.dot(X))**-1).dot(X.T)).dot(Y))
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = w.T.dot(Xs_test[i])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2)))
        mses.append(mse)
    return np.average(mses)


def poly_regression_cubic(X, Y, Xs_test, Ys_test):
    """
    This function performs third order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################

    X_poly = assemble_feature(X, 3)
    X_poly_test = assemble_feature(Xs_test, 3)

    w = ((((X.T.dot(X))**-1).dot(X.T)).dot(Y))
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = w.T.dot(Xs_test[i])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2)))
        mses.append(mse)
    return np.average(mses)


def neural_network(X_train, Y_train, Xs_test, Ys_test):
    """
    This function performs neural network prediction.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    d = 7

    ## YOUR CODE HERE
    #################
    learning_rate = 0.1
    iterations = 1000
    layer_1_nodes = 100
    layer_2_nodes = 100
    layer_3_nodes = 2

    #Setting up neural network
    X = tf.placeholder(tf.float32, shape=[None, d])
    Y = tf.placeholder(tf.float32, shape=[None, 2])

    # Setting up the neurons
    # Layer 1
    w1 = tf.Variable(tf.random_normal(shape=[7, layer_1_nodes],
                                               dtype=tf.float32,
                                               stddev=1e-1),
                                               name="weights1")
    b1 = tf.Variable(tf.constant(0.0, shape=[layer_1_nodes], dtype=tf.float32),
                                name="biases1")
    layer_1_output = tf.nn.relu(tf.matmul(X, w1) + b1)

    #Layer 2
    w2 = tf.Variable(tf.random_normal(shape=[layer_1_nodes, layer_2_nodes],
                                               dtype=tf.float32,
                                               stddev=1e-1),
                                               name="weights2")
    b2 = tf.Variable(tf.constant(0.0, shape=[layer_2_nodes], dtype=tf.float32), name="biases2")
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, w2) + b2)

    #Layer 3
    w3 = tf.Variable(tf.random_normal(shape=[layer_2_nodes, layer_3_nodes],
                                               dtype=tf.float32,
                                               stddev=1e-1),
                                               name="weights3")
    b3 = tf.Variable(tf.constant(0.0, shape=[layer_3_nodes], dtype=tf.float32), name="biases3")
    layer_3_output = tf.matmul(layer_2_output, w3) + b3

    #Optimization and training
    prediction = layer_3_output
    cost = tf.losses.absolute_difference(Y, prediction)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


    saver = tf.train.Saver()

    session = tf.InteractiveSession()

    session.run(tf.global_variables_initializer())

    for i in range(iterations):
        session.run(optimizer, feed_dict={X: X_train, Y: Y_train})

        if i % 20 == 0:
            training_cost = session.run(cost, feed_dict={X: X_train, Y:Y_train})
            current_prediction = session.run(prediction, feed_dict={X: X_train, Y:Y_train})
            accuracy = sum(current_prediction==Y_train) / len(Y_train)
            print("At iteration {}, the accuracy and cost are:".format(i))
            print("The accuracy on the training set is", accuracy)
            print("The current training cost: {}\n".format(training_cost))

    # Training is now complete!
    print("Training is complete!")

    final_training_cost = session.run(cost, feed_dict={X: X_train, Y:Y_train})
    final_testing_cost = session.run(cost, feed_dict={X: Xs_test, Y: Ys_test})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))
    saver.save(session, "my_net/model.ckpt")

    current_prediction = session.run(prediction, feed_dict={X: Xs_test, Y:Ys_test})
    accuracy = sum(current_prediction==Ys_test) / len(Ys_test)
    print("The testing accuracy is {}".format(accuracy))

    saver = tf.train.Saver()

    return final_testing_cost

# class HW6(object):
#
#     def __init__(self):
#         pass
if __name__ == '__main__':
    #Part a:
    #Test generative_model
    #Generate Data
    sensors = generate_sensors()
    distance, position = generate_dataset(sensors,
                         num_sensors=7,
                         spatial_dim=2,
                         num_data=1000,
                         original_dist=True,
                         noise=1)
    distance_test,position_test = generate_dataset(sensors,
                         num_sensors=7,
                         spatial_dim=2,
                         num_data=30,
                         original_dist=True,
                         noise=1)
    print (position)
    generative = generative_model(distance,position,distance_test,position_test)
    oracle = oracle_model (distance,position,distance_test,position_test, sensors)
    print (len(generative))
    print ("__________________")
    print (len(oracle))
    print ("__________________")
    # Model --> Linear Regression
    linear = linear_regression(distance,position,distance_test,position_test)
    print (len(linear))

    # # Model --> Second Order Polynomial Regression
    # poly = poly_regression_second(distance,position,distance_test,position_test)
    # print (poly)
    #
    # # Model --> Cubic Polynomial
    # cubic = poly_regression_cubic(distance,position,distance_test,position_test)
    # print (cubic)

    # # Model --> neural_network
    # neural = neural_network(distance,position,distance_test,position_test)
    # print (neural)
