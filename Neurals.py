import numpy as np

class NeuralNetwork():
    def __init__(self, layers : list | tuple, *args):
        '''
        layers : list -> Uses a list of undefined length to 
        figure out dimensions of the Neural Network 
        Example : [5,6,8] 
        This will create a 3 Layered Neural Network Creating :
        5 Neurons -> Input Layer
        6 Neurons -> 1st Hidden Layer
        8 Neurons -> Output Layer
        '''
        # TODO:Complete this tomorrow

        #Network Matrixes 
        self.neurons = np.random.rand(*layers)
        self.weights = self.create_weights(layers) 
        '''
        Weight between two layers can be represented as 2d matrix of NxM
        N -> no. of neurons in first layer
        M -> no. of neurons in second layer 

        So for completing the structure it will be a list of 2d matrices
        of dimensions NxM and length L
        L -> no. of layers - 1
        '''
        self.bias = [np.random.rand(layer) for layer in layers]

    def create_weights(self, layers: list|tuple):
        weight = []
        for i in range(len(layers)-1):
            weight.append(np.random.rand(layers[i],layers[i+1]))
        return weight

            

            
        
        
