import sys
sys.path.insert(0, '/Users/gently/Projects/ai-example/modules')
import numpy as np

from datasets.datasets import load
from modules.activations import Tanh, Sigmoid
from modules.dense import Dense
from modules.mean_squared_error import MeanSquaredError
from modules.softmax_cross_entropy import SoftmaxCrossEntropy
from modules.neural_network import NeuralNetwork
from modules.optimizer import SGD, SGDMomentum
from modules.trainer import Trainer


# init()
X_train, y_train, X_test, y_test = load()

# one-hot encode
num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1


# вычитания общего среднего
X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)

# деления на общую дисперсию
X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)

def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')

model = NeuralNetwork(
    layers=[Dense(neurons=89,
                  activation=Tanh(), weight_init="glorot", dropout=0.8),
            Dense(neurons=46,
                  activation=Tanh(),
                  weight_init="glorot",
                  dropout=0.8),
            Dense(neurons=10,
                  activation=Sigmoid(), weight_init="glorot")],
            loss = SoftmaxCrossEntropy(),
seed=20190119)

trainer = Trainer(model, SGDMomentum(lr = 0.1, momentum = 0.9, decay_type='exponential'))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=14881337,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)

