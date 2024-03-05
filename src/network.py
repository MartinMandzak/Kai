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
    weights connection from current layer and previous layer
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[1:], sizes[:-1])]

    """
    parameter "a" is simply an input of the network
    method returns network's output given a
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

        self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]

    """

    """



#class end
def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z))

#first derivative of sigmoid
def sigmoid_prime(z):
    return sigmoid(z) * (1- sigmoid(z))


net = Network([2,3,1])
