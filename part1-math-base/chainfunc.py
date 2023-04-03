
from typing import List, Callable
import numpy as np
from numpy import ndarray
from funcs import square

def chain_length_2(chain: List, x: ndarray) -> ndarray:
    '''
    Evaluates two functions in a row, in a "Chain".
    '''
    assert len(chain) == 2, \
    "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))

# A Chain is a list of functions
Chain = [square, square]

x = np.array([1,2,3])

print(chain_length_2(Chain, x))

