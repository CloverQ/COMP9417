import numpy as np
x = np.array([1, 1, 1, 1])
A = np.array([[1, 2, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b = np.array([3, 2, -2])
count = 0
while True:
    count += 1
    delta = (np.dot(np.dot(A.T, A), x)) - (np.dot(A.T, b)) + (0.2 * x)
    x = x - 0.1 * delta
    if np.dot(delta.T, delta) ** 0.5 < 0.001:
        break
    print(count)
    print(np.around(x, 4))

x_ = np.dot(np.linalg.pinv(np.dot(A.T, A) + 0.2*np.eye(4)), np.dot(A.T, b))
print(x_)

