import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray
# %matplotlib inline

from typing import Callable
from typing import Dict
from operation import Operation

class Linear(Operation):
    '''
    "Identity" activation function
    '''

    def __init__(self) -> None:
        '''Pass'''        
        super().__init__()

    def _output(self) -> ndarray:
        '''Pass through'''
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Pass through'''
        return output_grad
