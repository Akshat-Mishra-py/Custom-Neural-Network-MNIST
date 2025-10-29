import os 
import sys
#Appending another path so that we can import modules of in upper level of directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import unittest
from Neurals import NeuralNetwork

class TestNeuralClass(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.layers = [5,3,7]
        self.network = NeuralNetwork(self.layers)

    def test_neurons_dimensions(self):
        self.assertSequenceEqual(self.network.neurons.shape, self.layers)

    def test_weight_dimensions(self):
        self.assertEqual(len(self.network.weights), len(self.layers)-1)
        for i in range(len(self.layers)-1) :
            self.assertSequenceEqual(self.network.weights[i].shape, self.layers[i:i+2]) 

    def test_bias_shape(self):
        self.assertEqual(len(self.network.bias), len(self.layers))
        for i in range(len(self.layers)):
            self.assertEqual(*self.network.bias[i].shape, self.layers[i])


if __name__ == '__main__':
    unittest.main()