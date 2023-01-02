import numpy as np
import pandas as pd
import math
import warnings
import matplotlib.pyplot as plt


def initializing_x_y():
    np.random.seed(1)
    x = np.random.gamma(2, 0.1, 100)
    error = np.random.normal(loc=0, scale=1, size=100)
    beta0 = 100
    beta1 = 20
    beta2 = -50
    beta3 = 0.1
    y = beta0 + (beta1 * x) + (beta2 * x ** 2) + (beta3 * x ** 3 )+ error

    x0 = pd.DataFrame([1 for i in range(len(x))])
    x1 = pd.DataFrame(x)
    x2 = pd.DataFrame(x ** 2)
    x3 = pd.DataFrame(x ** 3)
    X = pd.concat([x0, x1, x2, x3], axis=1)
    return x, X, y

def Ridge_Reg(X, y):
    landa = 0.3
    reg_result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + np.identity(4) * landa), np.transpose(X)), y)
    return reg_result

def Standard_deviation(x):
    data_mean = x.mean()
    data_STD = x.std()
    normalized_data = (x - data_mean) / data_STD
    return normalized_data


def Ridge_normal(x, y):
    normalized_data = Standard_deviation(x)
    landa = 0.5
    beta0 = 100
    beta1 = 20
    beta2 = -50
    beta3 = 0.1
    ND0 = pd.DataFrame([1 for i in range(len(x))])
    ND1 = pd.DataFrame(normalized_data)
    ND2 = pd.DataFrame(normalized_data ** 2)
    ND3 = pd.DataFrame(normalized_data ** 3)
    normalized_data = pd.concat([ND0, ND1, ND2, ND3], axis=1)
    Reg_result_normal = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(normalized_data), normalized_data) + np.identity(4) * landa), np.transpose(normalized_data)), y)
    return normalized_data, Reg_result_normal

def Diagram():
    x, X, y = initializing_x_y()
    reg_result = Ridge_Reg(X, y)
    X_array = np.array(x)
    y_array = np.array(y)
    y_result_reg = (X.dot(reg_result))

    diagram_list_map = []
    for i in range(len(x)):
        pat = []
        pat.append(x[i])
        pat.append(y_result_reg[i])
        diagram_list_map.append(pat)
    array_sorted = sorted(diagram_list_map, key=lambda k: [k[0]])

    x_list = []
    y_list = []
    for i in range(len(x)):
        x_list.append(array_sorted[i][0])

    for i in range(len(x)):
        y_list.append(array_sorted[i][1])
    normalized_data, Reg_result_normal = Ridge_normal(x, y)
    diagram_list_map1 = (normalized_data.dot(Reg_result_normal))
    reg_result_norm1 = []
    for i in range(len(x)):
        mat = []
        mat.append(x[i])
        mat.append(diagram_list_map1[i])
        reg_result_norm1.append(mat)
    array_sorted_norm = sorted(reg_result_norm1, key=lambda k: [k[0]])
    x_norm = []
    y_norm = []
    for i in range(len(x)):
        x_norm.append(array_sorted_norm[i][0])

    for i in range(len(x)):
        y_norm.append(array_sorted_norm[i][1])

    plt.plot(x_list, y_list, label="Ridge_Reg")
    plt.plot(x_norm, y_norm, label="Normalized Ridge_Reg")
    plt.plot(x, y, 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

Diagram()
