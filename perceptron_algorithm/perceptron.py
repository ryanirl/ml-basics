from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


sample_size = 100

X, y = make_blobs(n_samples = sample_size, centers = 2, random_state = 1)
y[y == 0] = -1

print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")


# THE PERCEPTRON ALGORITHM 

def perceptron(t, D, y):
    th0s = np.array([0])
    ths = np.array([[0], [0]])

    for t in range(t):
        for i in range(sample_size):
            
            # this is unbelievably undreadable code oh boy.
            # I will be fixing this.
            if (y[i] * (np.dot(D[:, i], ths) + th0s) <= 0):
                ths = ths + np.transpose(np.expand_dims((y[i]) * D[:, i], axis = 0)) 
                th0s = th0s + y[i]

    return ths, th0s

# Calling Perceptron Algorithm
theta1, theta0 = perceptron(10000, X, y)

# PLOTTING POINTS

# Green for +1 and Blue for -1
for i in range(sample_size):
    if (y[i] == 1):
        plt.scatter(X[0][i], X[1][i], color="green") 
    else:
        plt.scatter(X[0][i], X[1][i], color="blue")

        
# *CHANGE NAMES* 
x = np.linspace(-10, 10, 10)
hyperplane = ((-1)*(theta1[0][0] / theta1[1][0]) * x) - (theta0[0]/theta1[1][0])

plt.plot(x, hyperplane, '-')

plt.show()

