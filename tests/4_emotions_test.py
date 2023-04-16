import sys

from sklearn.model_selection import train_test_split

sys.path.insert(0, '/Users/gently/Projects/ai-example/modules')
import numpy as np
from modules.activations import Tanh, Sigmoid, Linear
from modules.dense import Dense
from modules.softmax_cross_entropy import SoftmaxCrossEntropy
from modules.neural_network import NeuralNetwork
from modules.optimizer import SGDMomentum
from modules.trainer import Trainer

data_images = np.load('/Users/gently/Downloads/npdata/data.npy')
data_labels = np.load('/Users/gently/Downloads/npdata/label.npy')
image_array = []
for image_data in data_images:
    new_data = np.zeros((48, 144, 1), dtype=np.uint8)
    new_data[:, :48, 0] = image_data[:, :, 0]
    # Copy the green channel of the RGB data to the second third of the new array
    new_data[:, 48:96, 0] = image_data[:, :, 1]
    # Copy the blue channel of the RGB data to the last third of the new array
    new_data[:, 96:, 0] = image_data[:, :, 2]
    image_array.append(new_data.reshape(-1, 48*144))

x_params_train, x_params_test, y_ans_train, y_asn_test = train_test_split(np.squeeze(np.array(image_array)), data_labels, test_size=0.2, random_state=80718)


# вычитания общего среднего
x_params_train, x_params_test = x_params_train - np.mean(x_params_train), x_params_test - np.mean(x_params_train)

# деления на общую дисперсию
x_params_train, x_params_test = x_params_train / np.std(x_params_train), x_params_test / np.std(x_params_train)

def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), np.argmax(y_asn_test, axis=1)).sum() * 100.0 / test_set.shape[0]:.2f}%''')

model = NeuralNetwork(
    layers=[Dense(neurons=166,
                  activation=Tanh(), weight_init="glorot", dropout=0.8),
            Dense(neurons=82,
                  activation=Tanh(),
                  weight_init="glorot",
                  dropout=0.8),
            Dense(neurons=4,
                  activation=Linear(), weight_init="glorot")],
            loss = SoftmaxCrossEntropy(),
seed=14881337)

trainer = Trainer(model, SGDMomentum(lr = 0.01, momentum = 0.9, decay_type='exponential'))
trainer.fit(x_params_train, y_ans_train, x_params_test, y_asn_test,
            epochs = 50,
            eval_every = 10,
            seed=14881337,
            batch_size=60);
print()
calc_accuracy_model(model, x_params_test)

