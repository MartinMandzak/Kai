"""
Kai main body
"""

import network
import input_handler

training_data, validation_data, test_data = input_handler.load_data_wrapper()

Kai = network.Network([784,30,10])
# data, epochs, batch_size, eta, test_data
<<<<<<< HEAD
Kai.SGD(training_data, 5, 10, 2.5, test_data = test_data)
=======
Kai.SGD(training_data, 5, 10, 2.0, test_data = test_data)
>>>>>>> af57477762cdb21bc577bb0e7656548ecaf5b6f4
