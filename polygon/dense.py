
import numpy as np
from numpy import ndarray
from sigmoid import Sigmoid
from operation import Operation
from layer import Layer
from weight_multiply import WeightMultiply
from bias_add import BiasAdd

class Dense(Layer):
    '''
    A fully connected layer which inherits from "Layer"
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()):
        '''
        Requires an activation function upon initialization
        '''
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:
        '''
        Defines the operations of a fully connected layer.
        '''
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None