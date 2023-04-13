import random

import numpy as np
from sklearn.datasets import load_iris, load_sample_images
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from urllib import request
import gzip
import pickle

from modules.common import to_2d_array


def get_iris_data():
    boston = load_iris()
    data = boston.data
    target = boston.target

    s = StandardScaler()
    data = s.fit_transform(data)
    print(data.shape)
    print(target.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
    # make target 2d array
    y_train, y_test = to_2d_array(y_train), to_2d_array(y_test)
    return X_train, X_test, y_train, y_test

def get_numbers_data():

    data = []
    matrix_size = 1000
    for i in range(matrix_size):
        data.append([])
        for j in range(2):
            data[i].append(random.random())

    target = []
    for i in range(0, matrix_size):
        target.append((data[i][0]+data[i][1]))

    data = np.array(data)
    target = np.array(target)
    # commented for self testing
    s = StandardScaler()
    data = s.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
    # make target 2d array
    y_train, y_test = to_2d_array(y_train), to_2d_array(y_test)
    return X_train, X_test, y_train, y_test


filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]