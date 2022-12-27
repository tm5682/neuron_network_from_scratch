# import sys


# print ("Python:" , sys.version)
# print("Numpy:", np.__version__)
# print("Matplotlib:" , matplotlib.__version__)

# inputs =    [[1, 2, 3, 2.5 ],
#             [2.0, 5.0, -1.0, 2.0],
#             [-1.5, 2.7, 3.3, -0.8]]

# weights = [
#     [0.2, 0.8, -0.5, 1.0], 
#     [0.5, -0.91, 0.26, -0.5], 
#     [-0.26, -0.27, 0.17, 0.87]
# ]

# biases = [2,3, 0.5]


# # output = np.dot(weights, inputs) + biases
# # print (output)

# # we change weights to numpy array here to use transpose to switch rows and column 
# # else using .dot function numpy does it in the background

# output = np.dot(inputs, np.array(weights).T) + biases
# print (output)


import numpy as np
import matplotlib

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()



np.random.seed(0)

# X =    [[1, 2, 3, 2.5 ],
#         [2.0, 5.0, -1.0, 2.0],
#         [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data (100, 3)


class Layer_Dense:
    def __init__ (self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases =  np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Actiovation_ReLU:
    def forward (self, inputs):
        self.output = np.maximum(0,inputs)


layer1 = Layer_Dense(2,5)

activation1 = Actiovation_ReLU()

layer1.forward(X) 


activation1.forward(layer1.output)

print (activation1.output)

