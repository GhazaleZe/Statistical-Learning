import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import HW6_Q1
import HW6_Q1_b


def Folds_creation(x, y):
    fold_x_train = []
    fold_x_test = []
    fold_y_train = []
    fold_y = []

    fold_x_train.append(x[5:25])
    fold_y_train.append(y[5:25])
    fold_x_test.append(x[0:5])
    fold_y.append(y[0:5])
    # ***************************************************

    fold_x_train.append(np.vstack((x[0:5], x[10:25])))
    fold_y_train.append(np.concatenate((y[0:5], y[10:25])))
    # print(fold_y_train[1])
    fold_x_test.append(x[5:10])
    fold_y.append(y[5:10])
    # *****************************************************

    fold_x_train.append(np.vstack((x[0:10], x[15:25])))
    fold_y_train.append(np.concatenate((y[0:10], y[15:25])))
    fold_x_test.append(x[10:15])
    fold_y.append(y[10:15])
    # *****************************************************

    fold_x_train.append(np.vstack((x[0:15], x[20:25])))
    fold_y_train.append(np.concatenate((y[0:15], y[20:25])))
    fold_x_test.append(x[15:20])
    fold_y.append(y[15:20])
    # ****************************************************

    fold_x_train.append(x[0:20])
    fold_y_train.append(y[0:20])
    fold_x_test.append(x[20:25])
    fold_y.append(y[20:25])

    return np.array(fold_x_train), np.array(fold_x_test), np.array(fold_y), np.array(fold_y_train)


def Mean_Squared_Error(y, predicted):
    MSE = 0
    for i in range(len(y)):
        temp = (y[i] - predicted[i]) ** 2
        MSE += temp
    MSE = MSE / len(y)
    # print("MSE")
    # print(MSE)
    return MSE


def loss_run(x_train, x_test, y_train, y_test, lam):
    error = np.random.normal(loc=0, scale=1, size=20)
    final_list_errors_mean_in_k_fold = []
    error = []
    for k in range(len(lam)):
        error.clear()
        for number in range(5):  # each fold
            y_prediction = []
            w = HW6_Q1.initialize_w(k=101)
            alpha = HW6_Q1.largest_igenvalue(x_train[number])
            goal_w = HW6_Q1.weight_claculation(alpha, lam, k, x_train[number], y_train[number], w)
            value1 = 0
            for c in range(5):
                for o in range(101):
                    value1 += goal_w[o] * x_test[number][c][o]
                y_prediction.append(value1)
            error.append(Mean_Squared_Error(y_test[number], y_prediction))
        final_list_errors_mean_in_k_fold.append(sum(error) / len(error))
    return final_list_errors_mean_in_k_fold


def Rig_run(x_train, x_test, y_train, y_test, lam):
    error = np.random.normal(loc=0, scale=1, size=20)
    final_list_errors_mean_in_k_fold = []
    error = []
    for k in range(len(lam)):
        error.clear()
        for number in range(5):  # each fold
            y_prediction = []
            goal_w = HW6_Q1_b.Ridge_normal(x_train[number], y_train[number], lam, k)
            value1 = 0
            for c in range(5):
                for o in range(101):
                    value1 += goal_w[o] * x_test[number][c][o]
                y_prediction.append(value1)
            error.append(Mean_Squared_Error(y_test[number], y_prediction))
        final_list_errors_mean_in_k_fold.append(sum(error) / len(error))
    return final_list_errors_mean_in_k_fold


def main():
    x, y = HW6_Q1.read_data()
    train_x_fold, test_x_fold, y_fold_test, y_fold_train = Folds_creation(x, y)
    lambda_vector = np.linspace(0.01, 1, 10)

    error_list = Rig_run(train_x_fold, test_x_fold, y_fold_train, y_fold_test, lambda_vector)
    Rig_best_lambda = np.min(error_list)
    min_index = np.where(error_list == Rig_best_lambda)
    print("Best lamda for Ridge")
    print(lambda_vector[min_index])

    error_loss = loss_run(train_x_fold, test_x_fold, y_fold_train, y_fold_test, lambda_vector)
    loss_best_lamda = np.min(error_loss)
    min_index_loss = np.where(error_loss == loss_best_lamda)
    print("Best lamda for Losso, min error")
    print(lambda_vector[min_index_loss])



if __name__ == "__main__":
    main()
