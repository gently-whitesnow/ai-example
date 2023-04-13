import numpy as np
from numpy import ndarray

from loss import Loss

class MeanSquaredError(Loss):

    def __init__(self,
                 normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize

    def _output(self) -> float:

        if self.normalize:
            self.prediction = self.prediction / self.prediction.sum(axis=1, keepdims=True)

        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

        return loss

    def _input_grad(self) -> ndarray:

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]