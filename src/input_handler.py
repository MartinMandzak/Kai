"""
Code to parse MNIST dataset of handwritten digits...
god help me
"""

import pickle
import gzip
import numpy as np

"""
returns a vector with all zeroes except 1 which would represent the correct solution
e.g. (0,0,1,0,0,0,0,0,0,0) => correct answer is 2
"""
def vectorised_result(x):
    e = np.zeros((10,1))
    e[x] = 1.0
    return e

"""
This comment only exists because every other function has one and it drove me crazy
"""
def load_data():
    file = gzip.open("../data/mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = pickle.load(file, encoding = 'latin1')
    file.close()
    return (training_data, validation_data, test_data)


"""
more suitable format of load_data
"""
def load_data_wrapper():
    training_data, validation_data, test_data = load_data()

    #28x28px
    training_inputs = [np.reshape(x,(784,1)) for x in training_data[0]]
    training_outputs = [vectorised_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_outputs)

    validation_inputs = [np.reshape(x,(784,1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])

    test_inputs = [np.reshape(x,(784,1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])

    return (training_data, validation_data, test_data)
