import numpy

class Neuron():
    def __init__(self, activation_value : int):
        # Contains the activation value of the 
        self.activation_value = activation_value

    def calculateNumber(self, weight, bias):
        self.activation_value = weight * self.activation_value + bias

    def __add__(self, other):
        if type(other) == Neuron:
            return Neuron(self, self.activation_value + other.activation_value)
        elif type(other) in [int, float]:
            return Neuron(self, self.activation_value + other)


class NeuralNetwork():
    def __init__(self, layer : list, *args):
        pass
        

    
        
