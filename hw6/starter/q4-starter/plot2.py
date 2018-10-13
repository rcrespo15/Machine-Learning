import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from starter import *


def neural_network(X, Y, X_test, Y_test, num_neurons, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_neurons: number of neurons in each layer
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    d = 7
    X_train = X
    Y_train = Y
    ## YOUR CODE HERE
    #################
    learning_rate = 0.1
    iterations = 1000
    layer_1_nodes = num_neurons
    layer_2_nodes = num_neurons
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
    final_testing_cost = session.run(cost, feed_dict={X: X_test, Y: Y_test})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))
    saver.save(session, "my_net/model.ckpt")

    current_prediction = session.run(prediction, feed_dict={X: X_test, Y:Y_test})
    accuracy = sum(current_prediction==Y_test) / len(Y_test)
    print("The testing accuracy is {}".format(accuracy))

    saver = tf.train.Saver()

    return final_testing_cost
    ## YOUR CODE HERE
    #################


#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_neuronss = np.arange(100, 550, 50)
mses = np.zeros((len(num_neuronss), 2))

# for s in range(replicates):

sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_neurons in enumerate(num_neuronss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "ReLU")
    mses[t, 0] = mse

    # mse = neural_network(X, Y, X_test, Y_test, num_neurons, "tanh")
    # mses[t, 1] = mse

    print('Experiment with {} neurons done...'.format(num_neurons))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_neuronss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of neurons')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('num_neurons.png')
