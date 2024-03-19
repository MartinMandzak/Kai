"""
This is me rewriting the network.py file in order to get a deeper understanding of it
"""

import numpy as np
import random

class Network(object):
    """
    sizes is an array of integers
    in this instance sizes is 784 since its 28px^2 as the input layer; 30 as the hidden layer ;10 as the output layer
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        """
        randomly picking biases and weights before adjusting them as necessary
        biases generates a 2D array, where (y,1) means it'll generate array with the shape of y columns, 1 row
        weights 
        """
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[1:], sizes[:-1])]

    """
    a = activation value, it's being updated
    a´ = σ(wa+b)
    w,a,b are all vectors and so they must be multiplied like matrices
    """
    def feed_forward(self, activation):
        for w,b in zip(self.weights, self.biases):
            activation = sigmoid(np.dot(w,activation)+b)
        return activation
    """
    Stochastic gradient descent
    eta := learning rate
    mini_batches are created by randomly shuffling the training data and then splitting it by batch size
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        training_data = list(training_data)
        n_train = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[stop:mini_batch_size+stop] for stop in range(0,n_train,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print(f"Epoch {epoch+1} correct: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch+1} completed without evaluation")
    
    """
    literally backpropagation except we don't do it over the whole dataset since that would be computationally expensive
    """
    def update_mini_batch(self,mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for pixel_val,associated_num in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(pixel_val, associated_num)

            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        #update
        self.biases = [b - (eta/len(mini_batch))* nb for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta/len(mini_batch))* nw for w,nw in zip(self.weights, nabla_w)]

    """
    training_data = [x = pixel_values, y = associated_number]*
    first we create empty vectors which will be updated with new values
    then, we make pixel values our activation, throw it into the feed_forward and append
    then, we calculate the cost function, we use activations[:-1] to exclude the output layer
            and z_vectors[-1] to hit the layer before the current one
    then, we fill out the last element of nabla_b with our current delta
            and nabla_w gets the product of multiplying delta with activations[-2].transpose() <making rows into cols>
            we are multiplying delta with the second to last activation layer to adjust weights in the previous layer
    then, we go over each layer and update delta which is showing the error of the current layer
    then, this error is multiplied by delta of the next layer TIMES the derivative of sigmoid which just shows how sensitive it is to small changes
    then we update the current bias [-layer] and current weights <those are adjusted by a product of delta and transposed activation layer that came just before it
    DONE!!!!
    """
    def backprop(self, x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        z_vectors = []
        #feed_forward
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,activation) + b
            z_vectors.append(z)

            activation = sigmoid(z)
            activations.append(activation)
        
        #backprop
        delta  = self.cost_derivative(activations[:-1],y) * sigmoid_prime(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in (2,self.num_layers):
            z = z_vectors[-layer]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())

        return (nabla_b,nabla_w)


    """
    my_result - expected result ==> vector to move in
    """
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    """
    return the number of times network's guess was correct by comparing to test_data
    """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(guess_from_px)),associated_num) for (guess_from_px,associated_num) in test_data]
        return sum(int(guess_from_px == associated_num) for (guess_from_px,associated_num) in test_results)


#classend Network

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
