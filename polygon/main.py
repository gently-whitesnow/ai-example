import numpy as np
from sklearn.preprocessing import StandardScaler

from dense import Dense
from linear import Linear
from mean_squared_error import MeanSquaredError
from metrics import eval_regression_model
from neural_network import NeuralNetwork
from optimizer import SGD
from polygon.datasets import get_iris_data, get_numbers_data
from sigmoid import Sigmoid
from trainer import Trainer

X_train, X_test, y_train, y_test = get_numbers_data()
neural_network = NeuralNetwork(
    layers=[Dense(neurons=20,
                   activation=Sigmoid()),
            Dense(neurons=20,
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
print()
eval_regression_model(neural_network, X_test, y_test)
print("datafrom dataset")
for i in range(10):
    print(y_train[i])
print("my data")
test = [[0.1,0.7],[-0.4,0.6],[1.5,0.7],[2.0,0.1],[0.7,1.8]]
for i in test:
    print("test", i,"solve",i[0]+i[1])
print("ans")
pred = neural_network.forward(np.array(test))
print(pred)