import sys
sys.path.insert(0, '/Users/gently/Projects/ai-example/modules')

from modules.dense import Dense
from modules.linear import Linear
from modules.mean_squared_error import MeanSquaredError
from modules.estimate import eval_regression_model
from modules.neural_network import NeuralNetwork
from modules.optimizer import SGD
from datasets.datasets import get_iris_data
from modules.sigmoid import Sigmoid
from modules.trainer import Trainer

X_train, X_test, y_train, y_test = get_iris_data()
neural_network = NeuralNetwork(
    layers=[Dense(neurons=20,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

trainer = Trainer(neural_network, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=14881337)

eval_regression_model(neural_network, X_test, y_test)