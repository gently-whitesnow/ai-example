import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
