import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import *

def perceptron(X, y, max_iter = 100):
    np.random.seed(1)
    w = np.zeros(3)
    t = 1
    while t <= max_iter:
        misc_set = []
        for i in range(0, len(X)):
            if y[i] * (np.dot(w, X[i])) <= 0:
                misc_set.append(i)
        if len(misc_set) == 0:
            break
        idx = np.random.randint(0, len(misc_set))
        w = w + y[misc_set[idx]] * X[misc_set[idx]]
        t += 1
    return w, t-1

def dual_perceptron(X, y, max_iter = 100):
    np.random.seed(1)
    n = len(X)
    w = np.zeros(n)
    t = 1
    while t <= max_iter:
        misc_set = []
        for i in range(0, len(X)):
            sum = 0
            for j in range(0, len(X)):
                sum += y[j]*w[j] * np.dot(X[j], X[i])
            if y[i] * sum <= 0:
                misc_set.append(i)
        if len(misc_set) == 0:
            break
        idx = np.random.randint(0, len(misc_set))
        w[misc_set[idx]] = w[misc_set[idx]] + 1
        t += 1
    return w, t-1

def r_perceptron(X, y, max_iter = 100):
    np.random.seed(1)
    w = np.zeros(3)
    t = 1
    I = np.zeros(n)
    while t <= max_iter:
        misc_set = []
        for i in range(0, len(X)):
            if y[i] * (np.dot(w, X[i])) + 2 * I[i] <= 0:
                misc_set.append(i)
        if len(misc_set) == 0:
            break
        idx = np.random.randint(0, len(misc_set))
        w = w + y[misc_set[idx]] * X[misc_set[idx]]
        t += 1
        I[misc_set[idx]] = 1
    return w, t - 1

def r_dual_perceptron(X, y, max_iter = 100):
    np.random.seed(1)
    n = len(X)
    w = np.zeros(n)
    t = 1
    I = np.zeros(n)
    while t <= max_iter:
        misc_set = []
        for i in range(0, len(X)):
            sum = 0
            for j in range(0, len(X)):
                sum += y[j] * w[j] * np.dot(X[j], X[i])
            if y[i] * sum + 2 * I[i] <= 0:
                misc_set.append(i)
        if len(misc_set) == 0:
            break
        idx = np.random.randint(0, len(misc_set))
        w[misc_set[idx]] = w[misc_set[idx]] + 1
        t += 1
        I[misc_set[idx]] = 1
    return w, t - 1

X = pd.read_csv("PerceptronX.csv", header=None)
y = pd.read_csv("Perceptrony.csv", header=None)
X = np.array(X)
y = np.array(y)
w, nmb_iter = perceptron(X, y)
fig, ax = plt.subplots()
plot_perceptron(ax, X, y, w)
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name.png", dpi=300)
plt.show()

a, nmb_iter = dual_perceptron(X, y)
w = np.zeros(3)
y = y.flatten()
for i in range(0, len(X)):
    w = w + a[i] * X[i] * y[i]
fig, ax = plt.subplots()
plot_perceptron(ax, X, y, w)
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name.png", dpi=300)
plt.show()
n = len(X)
xx = np.arange(1, n+1)
plt.scatter(xx, a)
plt.show()

w, nmb_iter = r_perceptron(X, y)
print(w)
fig, ax = plt.subplots()
plot_perceptron(ax, X, y, w)
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name.png", dpi=300)
plt.show()

a, nmb_iter = r_dual_perceptron(X, y)
w = np.zeros(3)
y = y.flatten()
for i in range(0, len(X)):
    w = w + a[i] * X[i] * y[i]
fig, ax = plt.subplots()
plot_perceptron(ax, X, y, w)
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name.png", dpi=300)
plt.show()
n = len(X)
xx = np.arange(1, n+1)
plt.scatter(xx, a)
plt.show()



