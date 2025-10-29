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
        Weight between two layers can be represented as 2d matrix of N x M

        >>N -> no. of neurons in first layer
        >>M -> no. of neurons in second layer 


        So for completing the structure it will be a list of 2d matrices
        of dimensions NxM and length L
        L -> no. of layers - 1
        '''
        self.bias = [np.random.rand(layer) for layer in layers]

    def create_weights(self, layers: list|tuple) -> list[np.ndarray]:
        weight = []
        for i in range(len(layers)-1):
            weight.append(np.random.rand(layers[i],layers[i+1]))
        return weight

    def soft_max(self, Z : np.ndarray ) -> np.ndarray:
        denom = np.e ** Z
        denom = denom.sum()
        Z = (np.e ** Z )/denom 
        return Z

    def ReLU(self, Z: np.ndarray ) -> np.ndarray:
        indexes = np.where(Z < 0)
        for i in indexes:
            Z[i] = 0
        return Z    

    def forward_pass(self, activationfunction : dict = {}, customfunction=False):
        '''
        This function is forward pass of the neural network:

        * args:
            * activationfunction : dict

            >>This argument will only be used if customfunction = True.
            This argument is a dictionary of Layer number as keys 
            and functions as values which can be used  to find out 
            the new activation values of neurons.

            * customfunction : bool

            >>This argument is just a flag to use the costum defined 
            activation functions in the forward_pass

        Example:
        "if ReLU and SoftMax are fuctions defined by user then:"

        `self.forwardpass({1:ReLU,2:ReLU,3:SoftMax}, customfunction = True)`

        The functions like passed should take in a `numpy.ndarray` as parameters
        and return `numpy.ndarray` else the function will raise an exception.

        The layer of the Neural Network start with 0 -> Input layer and 
        all others are hidden layers other than last one which is 
        Output layer
        '''
        for i in range(1, len(self.neurons)):
            self.neurons[i] = self.weights[i]*self.neurons[i]+self.bias[i]

        if customfunction:
            for layer, function in activationfunction.items():
                self.neurons[layer] = function(self.neurons[layer])
                
        else:
            # We apply ReLU for hidden layers and SoftMax on last layer as a default
            for i in range(1,len(self.neurons)-1):
                self.neurons[i] = self.ReLU(self.neurons[i])
            self.neurons[-1] = self.soft_max(self.neurons[-1])