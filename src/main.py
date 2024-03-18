"""
Kai main body
"""

import network
import input_handler

training_data, validation_data, test_data = input_handler.load_data_wrapper()

Kai = network.Network([784,30,10])
Kai.SGD(training_data, 30, 10, 3.0, test_data = test_data)
