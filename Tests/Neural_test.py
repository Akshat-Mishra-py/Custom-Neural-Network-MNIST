import os 
import sys
#Appending another path so that we can import modules of in upper level of directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import unittest
from Neurals import NeuralNetwork, OneHot
import numpy as np

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
    
    def test_soft_max_algo(self):
        soft_max = self.network.soft_max(np.array([2,2]))
        self.assertSequenceEqual(list(soft_max), [0.5,0.5])
    
    def test_ReLU_algo(self):
        ReLU = self.network.ReLU(np.array([2,3,4,-7]))
        self.assertSequenceEqual(list(ReLU), [2,3,4,0])

class TestOneHot(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        element_table = [1,2,3,4,5,6,7,8,9]
        self.one_hot_obj = OneHot(element_table)
    
    def test_encode(self):
        self.assertSequenceEqual(list(self.one_hot_obj.encode(1)), [1,0,0,0,0,0,0,0,0]) 
    
    def test_decode(self):
        encoded_table = np.array([1,0,0,0,0,0,0,0,0])
        self.assertEqual(self.one_hot_obj.decode(encoded_table), 1)



if __name__ == '__main__':
    unittest.main()