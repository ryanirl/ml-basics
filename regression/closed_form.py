import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class LinearRegression:
    def __init__(self, regularization = "None", alpha = 0.01, reg_lambda = 0.001):
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.alpha = alpha

    def predict(self, w, X, b):
        return X.dot(w) + b

    def loss(self, pred, actual):
        MSE = np.sum(actual - pred) ** 2 
        reg = regularization_value[self.regularization](self.reg_lambda, self.weights)

        return (1.0 / self.m) * MSE + reg

    def fit(self, X, y):
        self.m, self.n = X.shape

        self.X = np.concatenate([X, np.ones((self.m, 1))], axis = 1)
        self.y = y

        self.w = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)

        self.weights = self.w[0]
        self.bias = self.w[1]


def main():
    n = 100

    # Cool trick Percy Liang used during one of his lectures for CS221. It 
    # generates fake Y values and if the program works then w and b will converge
    # to whatever you set the artificial weights/bias values too.
    artificial_weights = np.array([[-1]])
    artificial_bias = np.array([[3]])

    X = np.random.uniform(-2, 2, (n, 1))
    noise = np.random.uniform(-1, 1, (n, 1))

    y = X.dot(artificial_weights) + artificial_bias + (noise / 5.0)

    model = LinearRegression()
    model.fit(X, y)

    weight = model.weights[0]
    bias = model.bias[0]

    print("weight0: {} | bias: {}".format(weight, bias))

    plt.scatter(X, y, color = "blue")
    x = np.linspace(-2, 2, 2)
    hyperplane = weight * x + bias
    plt.plot(x, hyperplane, '-', color = "red")

    plt.show()


if __name__ == "__main__":
    main()





