import matplotlib.pyplot as plt
import numpy as np
A = np.array([[1, 2, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b = np.array([3, 2, -2])
alpha_set = np.array([0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.1, 0.15])
x_ = np.dot(np.linalg.pinv(np.dot(A.T, A) + 0.2*np.eye(4)), np.dot(A.T, b))
num = 0
for alpha in alpha_set:
    num += 1
    plt.subplot(3, 3, num)
    count = 0
    loss = np.array([])
    line = np.array([])
    x = np.array([1, 1, 1, 1])
    while count <= 10000:
        count += 1
        delta = (np.dot(np.dot(A.T, A), x)) - (np.dot(A.T, b)) + (0.2 * x)
        x = x - alpha * delta
        loss = np.append(loss, np.linalg.norm(x - x_))
        line = np.append(line, 0.001)
        if np.dot(delta.T, delta) ** 0.5 < 0.001:
            break
    step = np.arange(0, count)
    plt.plot(step, loss)
    plt.plot(step, line, color='red')
plt.show()



