import numpy as np
import sys
import math
from matplotlib import pyplot as plt
import time

np.random.seed(3141592)

# d_f(z) = dz/dx in terms of z = f(x)

def relu(z):
    return np.maximum(z, 0.0)

def d_relu(z):
    return np.where(z > 0, 1.0, 0.0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    return z * (1 - z)

class NeuralNetwork:

    """

    Parameters
    ----------

    batch_size: batch size for gradient descent

    features: number of features in the data, also the size of input layer

    architecture: list of hidden layer sizes

    target_classes: number of target classes, also the size of output layer
                    due to one-hot encoding

    activation: list of activation functions for each hidden, output layer

    """

    def __init__(self,
                 batch_size,
                 features,
                 architecture,
                 target_classes,
                 activation,
                 learning_rate,
                 eps=1e-4,
                 adaptive=False,
                 max_iter=1000):

        # indexing:
        # 0: input layer,
        # 1 - num_hidden_layers: hidden layers,
        # num_hidden_layers + 1: output

        # input validation
        assert len(activation) == len(architecture) + 1
        assert eps > 0
        assert batch_size > 0
        assert features > 0
        assert target_classes > 0
        assert learning_rate > 0
        assert max_iter > 0

        # architecture structure
        self.num_hidden_layers = len(architecture)
        self.features = features
        self.architecture = [features] + architecture + [target_classes] # changed
        self.target_classes = target_classes

        # activation functions and derivatives
        self.activation = [None for i in range(self.num_hidden_layers + 2)]
        self.d_activation = [None for i in range(self.num_hidden_layers + 2)]
        for i in range(len(activation)):
            if activation[i] == 'relu':
                self.activation[i + 1] = relu
                self.d_activation[i + 1] = d_relu
            elif activation[i] == 'sigmoid':
                self.activation[i + 1] = sigmoid
                self.d_activation[i + 1] = d_sigmoid
            else:
                raise ValueError('Unsupported activation function,'
                        'choose one of relu and sigmoid')

        # He initialization (variance of 2/(number of units in the previous layer))
        self.theta = [None] + [np.random.uniform(-1, 1, (n+1, k)) * math.sqrt(6/n) for (n, k) in zip(self.architecture[:-1], self.architecture[1:])]

        # SGD parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eps = eps
        self.adaptive = adaptive
        self.max_iter = max_iter

    def train(self, _x_train, _y_train):

        # reformatting data
        m = _x_train.shape[0]
        X_train = np.concatenate((np.ones((m, 1)), _x_train), axis=1)
        y_train = np.concatenate((np.ones((m, 1)), _y_train), axis=1)

        # variables to keep track of SGD
        prev_error = math.inf
        epoch = 1

        # for each layer, keep track of outputs of that layer
        # as well as the computed deltas
        layer_outputs = [None for _ in range(len(self.architecture))]
        delta = [None for _ in range(len(self.architecture))]

        while True:

            # max number of epochs - to prevent infinite loop
            # however this is never triggered in any of the runs
            if epoch == self.max_iter:
                break

            # choosing the learning rate
            learning_rate = self.learning_rate
            if self.adaptive:
                learning_rate /= math.sqrt(epoch)

            # shuffle X_train and y_train first
            p = np.random.permutation(m)
            X_train, y_train = X_train[p], y_train[p]

            # initialize variables related to SGD
            average_error = 0
            M = self.batch_size
            B = m // M

            for i in range(B):

                # extract mini-batch from the data
                input_batch_X = X_train[i * M : (i + 1) * M]
                input_batch_y = y_train[i * M : (i + 1) * M][:, 1:]

                # forward propagate and keep track of outputs of each unit
                layer_outputs[0] = input_batch_X
                for layer in range(1, len(self.architecture)):
                    layer_outputs[layer] = np.concatenate((np.ones((M, 1)), self.activation[layer](layer_outputs[layer - 1] @ self.theta[layer])), axis=1)
                last_output = layer_outputs[-1][:, 1:]
                last_d_activation = self.d_activation[-1]

                # compute loss
                average_error += np.sum((input_batch_y - last_output) ** 2) / (2 * M)

                # compute deltas using backpropagation
                delta[-1] = (input_batch_y - last_output).T * last_d_activation(last_output.T) / M
                for layer in range(len(self.architecture) - 2, 0, -1): # theta, layer_outputs
                    delta[layer] = (self.theta[layer + 1][1:, :] @ delta[layer + 1]) * self.d_activation[layer](layer_outputs[layer][:, 1:].T)

                # using deltas find gradient for each theta[layer] and
                # do batch update on theta
                for layer in range(1, len(self.architecture)):
                    self.theta[layer] += learning_rate * (delta[layer] @ layer_outputs[layer - 1]).T

            # average loss over this epoch
            average_error /= B
            #print('Iteration:', epoch, 'loss:', average_error)

            # main convergence criteria
            if abs(average_error - prev_error) < self.eps:
                return epoch, average_error

            prev_error = average_error
            epoch += 1

        return epoch, prev_error

    def predict(self, x_test):

        # reformatting for matching the data
        m = x_test.shape[0]
        layer_output = np.concatenate((np.array([np.ones(m)]).T, x_test), axis=1)

        # feedforwarding
        for layer in range(1, len(self.architecture)):
            layer_output = self.activation[layer](layer_output @ self.theta[layer])
            layer_output = np.concatenate((np.array([np.ones(m)]).T, layer_output), axis=1)

        # returning predictions as class labels (not one-hot encoding)
        return np.argmax(layer_output[:, 1:], axis=1)

def one_hot_encoder(y, num_classes):
    b = np.zeros((y.shape[0], num_classes))
    b[np.arange(y.shape[0]), y] = 1
    return b

def compressor(x):
    return np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

def mainB():

    # extracting data
    X_train, y_train = compressor(np.load(sys.argv[1])), np.load(sys.argv[2])
    X_test, y_test = compressor(np.load(sys.argv[3])), np.load(sys.argv[4])
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # statistics
    units = []
    test_accuracies = []
    train_accuracies = []
    elapsed_time = []

    # possible values for hidden layer units
    experimental_values = [1, 10, 50, 100, 500]

    # iterating over all choices for hidden layer units
    for hidden_layer_units in experimental_values:

        # parameters for the neural network
        num_hidden_layers = 1
        features = 784
        batch_size = 100
        activation = ['sigmoid' for i in range(num_hidden_layers + 1)]
        architecture = [hidden_layer_units] * num_hidden_layers
        target_classes = 10
        learning_rate = 0.1
        eps = 1e-4

        # initializing the neural network
        nn = NeuralNetwork(batch_size=batch_size,
                           features=features,
                           architecture=architecture,
                           target_classes=target_classes,
                           activation=activation,
                           learning_rate=learning_rate,
                           eps=eps)

        # training the data
        t = time.time()
        epoch, average_error = nn.train(X_train, one_hot_encoder(y_train, target_classes))

        # prediction on test and train data
        y_pred_test = nn.predict(X_test)
        y_pred_train = nn.predict(X_train)

        # statistics
        elapsed_time.append(time.time() - t)
        units.append(hidden_layer_units)
        test_accuracies.append(100 * y_pred_test[y_pred_test == y_test].shape[0] / y_pred_test.shape[0])
        train_accuracies.append(100 * y_pred_train[y_pred_train == y_train].shape[0] / y_pred_train.shape[0])

        # printing stats
        print('hidden layer units:', hidden_layer_units)
        print('test accuracy:', test_accuracies[-1], '%')
        print('train accuracy:', train_accuracies[-1], '%')
        print('time taken:', elapsed_time[-1])
        print('number of epochs:', epoch)
        print('average error:', average_error)

    # plotting the graphs
    plt.xscale('log')
    plt.title('Accuracy plot')
    plt.xlabel('Hidden layer units')
    plt.ylabel('Accuracy (in %)')
    plt.plot(units, test_accuracies, label='Test accuracies')
    plt.plot(units, train_accuracies, label='Train accuracies')
    plt.savefig('nn_accuracy_plot_nonadaptive.png')
    plt.close()
    plt.xscale('log')
    plt.title('Time taken')
    plt.xlabel('Hidden layer units')
    plt.ylabel('Time taken (in s)')
    plt.plot(units, elapsed_time)
    plt.savefig('nn_time_plot_nonadaptive.png')

def mainC():

    # extracting data
    X_train, y_train = compressor(np.load(sys.argv[1])), np.load(sys.argv[2])
    X_test, y_test = compressor(np.load(sys.argv[3])), np.load(sys.argv[4])
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # statistics
    units = []
    test_accuracies = []
    train_accuracies = []
    elapsed_time = []

    # possible values for hidden layer units
    experimental_values = [1, 10, 50, 100, 500]

    # common parameters for neural network
    num_hidden_layers = 1
    features = 784
    batch_size = 100
    activation = ['sigmoid' for i in range(num_hidden_layers + 1)]
    target_classes = 10
    learning_rate = 0.5
    eps = 1e-4

    # iterating over all hidden layer unit values
    for hidden_layer_units in experimental_values:

        # architecture
        architecture = [hidden_layer_units] * num_hidden_layers

        # initializing the neural network
        nn = NeuralNetwork(batch_size=batch_size,
                           features=features,
                           architecture=architecture,
                           target_classes=target_classes,
                           activation=activation,
                           learning_rate=learning_rate,
                           eps=eps,
                           adaptive=True)
        t = time.time()

        # training
        epoch, average_error = nn.train(np.copy(X_train), one_hot_encoder(y_train, target_classes))

        # prediction on test and train data
        y_pred_test = nn.predict(np.copy(X_test))
        y_pred_train = nn.predict(np.copy(X_train))

        # statistics
        elapsed_time.append(time.time() - t)
        units.append(hidden_layer_units)
        test_accuracies.append(100 * y_pred_test[y_pred_test == y_test].shape[0] / y_pred_test.shape[0])
        train_accuracies.append(100 * y_pred_train[y_pred_train == y_train].shape[0] / y_pred_train.shape[0])

        # printing statistics
        print('hidden layer units:', hidden_layer_units)
        print('test accuracy:', test_accuracies[-1], '%')
        print('train accuracy:', train_accuracies[-1], '%')
        print('time taken:', elapsed_time[-1])
        print('number of epochs:', epoch)
        print('average error:', average_error)

    # plotting
    plt.xscale('log')
    plt.title('Accuracy plot')
    plt.xlabel('Hidden layer units')
    plt.ylabel('Accuracy (in %)')
    plt.plot(units, test_accuracies, label='Test accuracies')
    plt.plot(units, train_accuracies, label='Train accuracies')
    plt.savefig('nn_accuracy_plot_adaptive.png')
    plt.close()
    plt.xscale('log')
    plt.title('Time taken')
    plt.xlabel('Hidden layer units')
    plt.ylabel('Time taken (in s)')
    plt.plot(units, elapsed_time)
    plt.savefig('nn_time_plot_adaptive.png')

def mainD():

    # extracting data
    X_train, y_train = compressor(np.load(sys.argv[1])), np.load(sys.argv[2])
    X_test, y_test = compressor(np.load(sys.argv[3])), np.load(sys.argv[4])
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # statistics
    units = []
    test_accuracies = []
    train_accuracies = []
    elapsed_time = []

    # parameters for the neural networks
    num_hidden_layers = 2
    hidden_layer_units = 100
    architecture = [hidden_layer_units] * num_hidden_layers
    features = 784
    batch_size = 100
    relu_activation = ['relu' for i in range(num_hidden_layers + 1)]
    relu_activation[-1] = 'sigmoid'
    sigmoid_activation = ['sigmoid' for i in range(num_hidden_layers + 1)]
    target_classes = 10
    learning_rate = 0.5
    eps = 1e-4

    # iterating over both architectures
    for activation in [relu_activation, sigmoid_activation]:

        # initializing the neural network
        nn = NeuralNetwork(batch_size=batch_size,
                           features=features,
                           architecture=architecture,
                           target_classes=target_classes,
                           activation=activation,
                           learning_rate=learning_rate,
                           eps=eps,
                           adaptive=True)
        t = time.time()

        # training
        epoch, average_error = nn.train(np.copy(X_train), one_hot_encoder(y_train, target_classes))

        # prediction on test and training data
        y_pred_test = nn.predict(np.copy(X_test))
        y_pred_train = nn.predict(np.copy(X_train))

        # statistics
        elapsed_time.append(time.time() - t)
        units.append(hidden_layer_units)
        test_accuracies.append(100 * y_pred_test[y_pred_test == y_test].shape[0] / y_pred_test.shape[0])
        train_accuracies.append(100 * y_pred_train[y_pred_train == y_train].shape[0] / y_pred_train.shape[0])

        # printing statistics
        if activation == relu_activation:
            print('relu')
        else:
            print('sigmoid')
        print('hidden layer units:', hidden_layer_units)
        print('test accuracy:', test_accuracies[-1], '%')
        print('train accuracy:', train_accuracies[-1], '%')
        print('time taken:', elapsed_time[-1])
        print('number of epochs:', epoch)
        print('average error:', average_error)

def mainE():

    # extracting data
    X_train, y_train = compressor(np.load(sys.argv[1])), np.load(sys.argv[2])
    X_test, y_test = compressor(np.load(sys.argv[3])), np.load(sys.argv[4])
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # statistics
    units = []
    test_accuracies = []
    train_accuracies = []
    elapsed_time = []

    # parameters for the neural network
    batch_size = 100
    experimental_values = [(100, 100)]
    activation_choices = ['relu', 'logistic']

    # iterating over all possible experimental values
    for activation in activation_choices:

        from sklearn.neural_network import MLPClassifier
        hidden_layer_units = experimental_values[0]

        print(activation)

        # initializing the neural network
        nn = MLPClassifier(activation=activation,
                           hidden_layer_sizes=hidden_layer_units,
                           solver='sgd',
                           batch_size=batch_size,
                           learning_rate='invscaling',
                           learning_rate_init=0.5,
                           max_iter=1000,
                           nesterovs_momentum=False,
                           verbose=False,
                           random_state=3141592,
                           n_iter_no_change=1)
        t = time.time()

        # training the neural network
        nn.fit(np.copy(X_train), y_train)

        # prediction on training and test data
        y_pred_test = nn.predict(np.copy(X_test))
        y_pred_train = nn.predict(np.copy(X_train))

        # statistics
        elapsed_time.append(time.time() - t)
        units.append(hidden_layer_units)
        test_accuracies.append(100 * y_pred_test[y_pred_test == y_test].shape[0] / y_pred_test.shape[0])
        train_accuracies.append(100 * y_pred_train[y_pred_train == y_train].shape[0] / y_pred_train.shape[0])

        # printing statistics
        print('hidden layer units:', hidden_layer_units)
        print('test accuracy:', test_accuracies[-1], '%')
        print('train accuracy:', train_accuracies[-1], '%')
        print('time taken:', elapsed_time[-1])

def demo():
    part = sys.argv[5].lower()
    if part == 'b':
        mainB()
    elif part == 'c':
        mainC()
    elif part == 'd':
        mainD()
    elif part == 'e':
        mainE()

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def main():

    # extracting data
    X_train, y_train = compressor(np.load(sys.argv[1])), np.load(sys.argv[2])
    X_test = compressor(np.load(sys.argv[3]))
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # output file
    output_file = sys.argv[4]
    batch_size = int(sys.argv[5])
    hidden_layer_list = [int(i) for i in sys.argv[6].split()]
    given_activation = sys.argv[7]

    # parameters for the neural network
    num_hidden_layers = len(hidden_layer_list)

    features = X_train.shape[1]
    target_classes = y_train.max() + 1

    activation = ['sigmoid' for i in range(num_hidden_layers + 1)]
    if given_activation == 'relu':
        for i in range(num_hidden_layers):
            activation[i] = 'relu'

    architecture = hidden_layer_list
    learning_rate = 0.1
    eps = 1e-4

    #print(batch_size)
    #print(features)
    #print(architecture)
    #print(target_classes)
    #print(activation)
    #print(learning_rate)
    #print(eps)

    # initializing the neural network
    nn = NeuralNetwork(batch_size=batch_size,
                       features=features,
                       architecture=architecture,
                       target_classes=target_classes,
                       activation=activation,
                       learning_rate=learning_rate,
                       eps=eps,
                       adaptive=False)

    # training the model
    epoch, average_error = nn.train(X_train, one_hot_encoder(y_train, target_classes))

    # prediction on test and train data
    y_pred_test = nn.predict(X_test)

    write_predictions(output_file, y_pred_test)
    pass

if __name__ == '__main__':
    main()
