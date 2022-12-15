from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
X = pd.read_csv('CarSeats.csv', usecols=[1, 2, 3, 4, 5, 7, 8])
Y = pd.read_csv('CarSeats.csv', usecols=[0])
X = X.values
Y = Y.values
X_scaler = StandardScaler().fit(X)
print("mean: ", X_scaler.mean_)
print("variance: ", X_scaler.var_)
scaler_X = X_scaler.transform(X)
X_train = scaler_X[:200]
X_test = scaler_X[200:400]
Y_train = Y[:200]
Y_test = Y[200:400]
print("X_train[0]: ", X_train[0])
print("X_train[-1]: ", X_train[-1])
print("Y_train[0]: ", Y_train[0])
print("Y_train[-1]: ", Y_train[-1])
print("X_test[0]: ", X_test[0])
print("X_test[-1]: ", X_test[-1])
print("Y_test[0]: ", Y_test[0])
print("Y_test[-1]: ", Y_test[-1])

def L(x, y, beta):
    m, n = x.shape
    ans = (1/m) * np.dot((y-np.dot(x, beta)).T, (y-np.dot(x, beta))) + 0.5*np.dot(beta.T, beta)
    return ans

def BGD(x, y, beta_R, x_t, y_t, max_count = 1000):
    m, n = x.shape
    alpha_set = np.array([0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])
    num = 0
    for alpha in alpha_set:
        num += 1
        plt.subplot(3, 3, num)
        beta = np.ones((n,), dtype=np.float64)
        count = 0
        loss = np.array([])
        while count < max_count:
            count += 1
            delta = np.zeros((n,), dtype=np.float64)
            for i in range(m):
              delta += -2*x[i] * (y[i]-np.dot(x[i].T, beta)) + beta
            beta = beta - (alpha/m) * delta
            loss = np.append(loss, L(x, y, beta) - L(x, y, beta_R))
        if alpha == 0.01:
            MSE1 = y - np.dot(x, beta)
            Train_MSE = (1/m) * (np.linalg.norm(MSE1) ** 2)
            MSE2 = y_t - np.dot(x_t, beta)
            Test_MSE = (1/m) * (np.linalg.norm(MSE2) ** 2)
            print("Train_MSE: ", Train_MSE)
            print("Test_MSE: ", Test_MSE)
        step = np.arange(0, 1000)
        plt.plot(step, loss)
    plt.show()

def Compute(x, y):
    m, n = x.shape
    I = np.eye(n)
    beta_R = np.dot(np.linalg.pinv(np.dot(x.T, x) + m*0.5*I), np.dot(x.T, y))
    return beta_R

def SGD(x, y, beta_R, x_t, y_t, max_count = 1000):
    m, n = x.shape
    alpha_set = np.array([0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02])
    num = 0
    for alpha in alpha_set:
        num += 1
        plt.subplot(3, 3, num)
        beta = np.ones((n,), dtype=np.float64)
        count = 0
        loss = np.array([])
        while count < max_count:
            count += 1
            delta = np.zeros((n,), dtype=np.float64)
            i = random.randint(0, m - 1)  # 随机选择一个样本
            delta += -2*x[i] * (y[i]-np.dot(x[i].T, beta)) + beta
            beta = beta - alpha * delta
            loss = np.append(loss, L(x, y, beta) - L(x, y, beta_R))
        if alpha == 0.001:
            MSE1 = y - np.dot(x, beta)
            Train_MSE = (1/m) * (np.linalg.norm(MSE1) ** 2)
            MSE2 = y_t - np.dot(x_t, beta)
            Test_MSE = (1/m) * (np.linalg.norm(MSE2) ** 2)
            print("Train_MSE: ", Train_MSE)
            print("Test_MSE: ", Test_MSE)
        step = np.arange(0, 1000)
        plt.plot(step, loss)
    plt.show()

Y_train = Y_train.flatten()
beta_R = Compute(X_train, Y_train)
print("beta_R = ", beta_R)
BGD(X_train, Y_train, beta_R, X_test, Y_test)
SGD(X_train, Y_train, beta_R, X_test, Y_test)
