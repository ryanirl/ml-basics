import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# Neat trick I use to allow user string imput without adding multiple
# if else statements. These values simply allow someone to say if they
# want to use l1, l2, or no regularization.
regularization_value = defaultdict(lambda: "ERROR")
regularization_grad = defaultdict(lambda: "ERROR")

regularization_value["None"] = (lambda alpha, theta: 0)
regularization_value["l1"] = (lambda alpha, theta: alpha * np.sum(np.abs(theta)))
regularization_value["l2"] = (lambda alpha, theta: alpha * np.sum(theta ** 2))

regularization_grad["None"] = (lambda alpha, theta: 0)
regularization_grad["l1"] = (lambda alpha, theta: alpha * np.sum(np.sign(theta)))
regularization_grad["l2"] = (lambda alpha, theta: alpha * 2 * np.sum(theta))


class LinearRegression:
    def __init__(self, regularization = "None", alpha = 0.01, reg_lambda = 0.001):
        """
        Potential Regularization terms are:
            - l1 (Lasso Regression)
            - l2 (Ridge Regression)
            - None (Non-Regularized OLS Regression)

        """
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

        self.weights = np.random.uniform(-1, 1, (self.n, 1))
        self.bias = np.random.uniform(-1, 1, (1, 1))

        for i in range(1000):
            reg_grad = regularization_grad[self.regularization](self.reg_lambda, self.weights)
            pred = self.predict(self.weights, X, self.bias)

            if i % 100 == 0: print("Loss at epoch {} is: {}".format(i, self.loss(pred, y)))

            self.weights = self.weights - (self.alpha * 2 * (1.0 / self.m) * X.T.dot(pred - y)) + reg_grad
            self.bias = self.bias - (self.alpha * 2 * (1.0 / self.m) * np.sum(pred - y)) 



    

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





