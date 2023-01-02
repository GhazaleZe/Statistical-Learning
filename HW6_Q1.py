import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt


def initialize_w(k):
    w = np.ones(k)
    return w


def read_data():
    df = pd.read_csv("sparse_reg.csv")
    y_data = df.iloc[:, 100]
    x_data = df.iloc[:, 0:100]
    x_data0 = pd.DataFrame([1 for i in range(25)])
    x_data_f = pd.concat([x_data0, x_data], axis=1)
    x = np.array(x_data_f)
    y = np.array(y_data)
    return x, y


def largest_igenvalue(x):
    Xt = x.transpose()
    XtX = np.dot(Xt, x)
    w, v = eig(XtX)
    alpha = np.max(w)
    alpha_real = np.real(alpha)
    return alpha_real


def weight_claculation(alpha, lamda,lamda_index , x, y, w):
    weights = [w]
    for i in range(len(y)):
        XWs = np.dot(x, np.array(weights[i]))
        yXWs = y - XWs
        x_transpose = x.transpose()
        temp = np.dot(x_transpose, yXWs)
        q = ((1 / alpha) * temp) + np.array(weights[i])
        sigma = (lamda[lamda_index]/(2*alpha))
        w_temp = np.sign(q) * np.maximum(0, np.abs(q) - sigma)
        weights.append(w_temp)
    return weights.pop()

def plotting():
    lambda_ = [0.01, 0.1, 1, 10]
    k = 101
    x_index = np.arange(0, k, 1, dtype=int)
    X, y = read_data()
    # print(X)
    alpha = largest_igenvalue(X)

    w = initialize_w(k=k)
    # for x in range(k):
    goal_w = []
    for i in range(len(lambda_)):
        temp_w = weight_claculation(alpha, lambda_, i, X, y, w)
        goal_w.append(temp_w)

    plt.plot(x_index, goal_w[0], 'b', label='goal_w=0.01')
    plt.plot(x_index, goal_w[1], 'r', label='goal_w=0.1')
    plt.plot(x_index, goal_w[2], 'y', label='goal_w=1')
    plt.plot(x_index, goal_w[3], 'c', label='goal_w=10')
    plt.stem(x_index, goal_w[3])
    plt.stem(x_index, goal_w[2])
    plt.stem(x_index, goal_w[1])
    plt.stem(x_index, goal_w[0])

    (markers, stemlines, baseline) = plt.stem(goal_w[3])
    (markers, stemlines, baseline) = plt.stem(goal_w[2])
    (markers, stemlines, baseline) = plt.stem(goal_w[1])
    (markers, stemlines, baseline) = plt.stem(goal_w[0])
    plt.setp(markers, marker='o', markersize=10, markeredgecolor="orange", markeredgewidth=2)
    plt.legend()
    plt.show()


def main():
    x = 1


if __name__ == "__main__":
    main()
