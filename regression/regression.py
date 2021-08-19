import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_linear_regression
from sklearn.datasets import make_regression


def hypothesis(w, x, b):
    return np.dot(w.T, x) + b


# Ordinary Least Squares Regression
def OLS(X, y, d):
    w = np.array([[0]] * (d - 1))
    b = np.array([[0]])

    for i in range(100000):
        h = hypothesis(w, X, b)

        w = w - ( 0.01 * (2.0 * np.dot(h - y, X.T) / 200.0))
        b = b - ( 0.01 * (2.0 * np.sum(h - y)) / 200.0 )
        # print("w: {}, b: {}".format(w, b))
        
    return w, b


def L1_cost():
    pass

def lassoL1():
    pass

def ridgeL2():
    pass

def polynomial():
    pass

def bayes_lin():
    pass
    


class Regression:
    def __init__(self, kind = OLS):
        self.kind = kind

    def fit(X, y):
        pass

    def predict(X):
        pass

    

def main():
    n = 200

    X = (4) * np.random.rand(n)[np.newaxis] - 2 # gives values [-2, 2)


    # Cool trick Percy Liang used during one of his lectures for CS221. It 
    # generates fake Y values and if the program works then w and b will converge
    # to whatever you set the artificial weights/bias values too.
    artificial_weights = np.array([[-1]])
    artificial_bias = np.array([[3]])

    noise = np.random.rand(n)
    Y = np.dot(artificial_weights.T, X) + artificial_bias + (noise / 5.0)

    w, b = OLS(X, Y, 2)
    print(w)
    print(b)

    plt.scatter(X, Y, color="blue")
    x = np.linspace(-2, 2, 2)
    hyperplane = w[0] * x + b[0]

    plt.plot(x, hyperplane, '-', color="red")

    plt.show()




if __name__ == "__main__":
    main()
