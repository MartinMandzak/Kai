""""
Kai learning on MNIST numbers... this can only go poorly lmao
following the http://neuralnetworksanddeeplearning.com
thanks
"""

import numpy as np
import random

class Network(object):
    """
    sizes = num of neurons in each layer
    so [2,3,4,5] would be 4 layers of 2,3,4,5 neurons respectively with the first layer being the input layer
    biases = neurons in each layer
    weights = connection from current layer and previous layer
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[1:], sizes[:-1])]

    """
    parameter "a" is simply an input of the network
    np.dot == matrix multiplication
    """
    def feed_forward(self,a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    """
    Stochastic gradient descent
    a fairly simple method which shuffles training data, creates batches and calls the update method
    update method completes 1 step of gradient descent
    then simply prints out current progress
    """
    def SGD(self, training_data,epochs,mini_batch_size,eta,test_data=None):
        training_data = list(training_data)
        test_data = list(test_data)

        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    """
    updates mini batches by making them undergo 1 step of gradient descent using backpropagation
    "mini_batch" is a list of tuples "(x, y)", and "eta" is the learning rate
    ∇ == nabla
    """
    def update_mini_batch(self,mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    """
    Backpropagation, essentially a big fuckoff vector of sums of adjusted weights for our lovely little network to learn
    """
    def backprop(self, x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    #feed_forward
        activation = x
        activations = [x]
        z_vectors = []

        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #weight * activation_vector + bias
            z_vectors.append(z)

            activation = sigmoid(z) #!!!
            activations.append(activation)
    #backpass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # rows := cols
        
        for L in range(2,self.num_layers):
            z = z_vectors[-L]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-L+1].transpose(), delta) * sp
            nabla_b[-L] = delta
            nabla_w[-L] = np.dot(delta, activations[-L-1].transpose())

        return (nabla_b,nabla_w)

    """
    find the highest level of activation
    """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)),y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
    
    """
    Fuck me.
    """
    def cost_derivative(self, output_activations,y):
        return (output_activations-y)

#Network class end
def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z))

#first derivative of sigmoid
def sigmoid_prime(z):
    return sigmoid(z) * (1- sigmoid(z))

