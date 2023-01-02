import numpy as np
import matplotlib.pyplot as plt
import math
import warnings


def X_Vectore(X):
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX


def Sigmoid(X):
    value = 1 / (1 + math.e ** (-X))
    return value


# initial values of vector 0
def initialize_theta(X):
    weight = np.random.randn(len(X[0]) + 1, 1)
    return weight


def My_Logistics_Regression(X, y, rate, iterations_number):
    y_temp = np.reshape(y, (len(y), 1))
    vector_X = X_Vectore(X)
    weight = initialize_theta(X)
    len_X = len(X)
    gradients = 0
    sum_final = 0
    Cost_values = []
    for i in range(iterations_number):
        gradients = 1 / len_X * vector_X.T.dot(Sigmoid(vector_X.dot(weight)))
        weight = weight - (rate * gradients)
    return weight


def read_dataset(dataset):
    text_file = open(dataset, 'r').read().splitlines()
    lines = text_file
    data = []
    temp = []
    Y = []
    for i in range(len(lines)):
        temp = lines[i].split(",")
        Y.append(int(temp[2]))
        temp = temp[:-1]
        X = list(map(float, temp))
        data.append(X)
    X_train = np.array(data)
    Y_train = np.array(Y)
    Y_train[Y_train == 0] = -1
    return X_train, Y_train


warnings.filterwarnings('ignore')
X_train, Y_train = read_dataset("logistic-data.txt")
#print(X_train)
#print(Y_train)

Coefficients = My_Logistics_Regression(X_train, Y_train, 0.001, 10000)
print(Coefficients)
