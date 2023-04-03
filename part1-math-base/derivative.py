import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray

from funcs import square, leaky_relu, sqrt

from typing import Callable

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          diff: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the "input_" array.
    '''
    return (func(input_ + diff) - func(input_ - diff)) / (2 * diff)

a = np.array([1,2,3])

print(deriv(square,a))

a = np.array([4,9,16])
print(deriv(sqrt,a))