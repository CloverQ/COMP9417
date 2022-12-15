import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy

def reg_log_loss(W, C, X, y):
    w = W[1:]
    c = W[0]
    term1 = 0.5 * np.dot(w.T, w)
    sum = 0.0
    n = len(X)
    for i in range(0, n):
        sum += np.logaddexp(0, -y[i]*(np.dot(w.T, X[i]) + c))
    term2 = sum * C
    return term1 + term2

def reg_log_fit(X, y, C):
    w = 0.1 * np.ones(X.shape[1])
    W0 = np.insert(w, 0, -1.4)
    g = lambda x: reg_log_loss(x, C, X, y)
    res = minimize(g, W0, method='Nelder-Mead', tol=1e-6)
    return res.x

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

X_ = pd.read_csv('songs.csv', usecols=[2, 3, 4, 6, 8, 9, 11, 12, 13, 14])
Y_ = pd.read_csv('songs.csv', usecols=[16])
X_ = X_.values
Y_ = Y_.values
Y_ = Y_.flatten()
n = len(X_)
X = []
Y = []
for i in range(0, n):
    if np.isnan(X_[i]).any():
        continue
    if Y_[i] == 5:
        X.append(X_[i])
        Y.append(1)
    elif Y_[i] == 9:
        X.append(X_[i])
        Y.append(-1)
X = np.array(X)
Y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=23)
x_tmp = X_train
scl = StandardScaler().fit(x_tmp)
X_train = scl.transform(X_train)
X_test = scl.transform(X_test)
print("X_train first row: ", X_train[0][0], X_train[0][1], X_train[0][2])
print("X_train last row: ", X_train[-1][0], X_train[-1][1], X_train[-1][2])
print("X_test first row: ", X_test[0][0], X_test[0][1], X_test[0][2])
print("X_test last row: ", X_test[-1][0], X_test[-1][1], X_test[-1][2])
print("y_train first row: ", y_train[0])
print("y_train last row: ", y_train[-1])
print("y_test first row: ", y_test[0])
print("y_test last row: ", y_test[-1])

c = 1.2
w = 0.35 * np.ones(X_train.shape[1])
W = np.insert(w, 0, c)
loss = reg_log_loss(W = W, C = 0.001, X = X_train, y = y_train)
print(loss)

res_x = reg_log_fit(X_train, y_train, C = 0.4)
w_pre = res_x[1:]
c_pre = res_x[0]
y_pre = []
n = len(X_train)
for i in range(0, n):
    y_pre.append(sigmoid(np.dot(w_pre.T, X_train[i]) + c_pre))
y_pre = np.array(y_pre)
LogLoss = log_loss(y_train, y_pre)
print("train and test loss of resulting model: ", LogLoss)
clf = LogisticRegression(C=1, tol=1e-6, penalty='l2', solver='liblinear')
clf.fit(X_train, y_train)
y_pre_clf = clf.predict(X_train)
LogLoss_clf = log_loss(y_train, y_pre_clf)
print("train and test loss of logistic regression model: ", LogLoss_clf)

Cs = np.linspace(0.01, 0.2, num=100)
num = len(Cs)
coef = []
for i in range(0, num):
    clf_g = LogisticRegression(penalty='l1', solver='liblinear', C=Cs[i])
    clf_g.fit(X_train, y_train)
    length = len(clf_g.coef_[0])
    tmp = []
    for j in range(0, length):
        tmp.append(clf_g.coef_[0][j])
    tmp = np.array(tmp)
    coef.append(tmp)
coef = np.array(coef)
print(coef)
dic = {0:"red", 1:"brown", 2:"green", 3:"blue", 4:"orange", 5:"pink", 6:"purple", 7:"grey", 8:"black", 9:"y"}
for i in range(0, 10):
    tmp = []
    for j in range(0, num):
        tmp.append(coef[j][i])
    tmp = np.array(tmp)
    plt.plot(Cs, tmp, color=dic[i])
plt.show()

X_train_20 = X_train[:544]
y_train_20 = y_train[:544]
Cs = np.linspace(0.0001, 0.8, num = 25)
LogLoss_C = []
for k in range(0, len(Cs)):
    sum = 0.0
    for i in range(0, 544):
        X_train_i = np.delete(X_train_20, i, 0)
        y_train_i = np.delete(y_train_20, i, 0)
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=Cs[k])
        clf.fit(X_train_i, y_train_i)
        y_pre_i = clf.predict(X_train_i)
        LogLoss_i = log_loss(y_train_i, y_pre_i)
        sum += LogLoss_i
    avg = sum / 544
    LogLoss_C.append(avg)
LogLoss_C = np.array(LogLoss_C)
plt.plot(Cs, LogLoss_C)
plt.show()


