import numpy as np
from numpy import ndarray
from scipy.special import logsumexp

def to_2d_array(a: ndarray,
                type: str = "col") -> ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
        "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)
def normalize(a: np.ndarray):
    other = 1 - a
    return np.concatenate([a, other], axis=1)

def unnormalize(a: np.ndarray):
    return a[np.newaxis, 0]

def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))