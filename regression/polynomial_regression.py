import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

n = 200
degree = 3

X = np.random.uniform(-2, 2, (n, 1))

actificial_data = np.zeros((n, degree))

for i in range(degree):
    actificial_data[:, i] = (X ** (i + 1))[:, 0]


# Cool trick Percy Liang used during one of his lectures for CS221. It 
# generates fake Y values and if the program works then w and b will converge
# to whatever you set these values too.
artificial_weights = np.array([[0.9], [0.6], [0.3]])

noise = np.random.uniform(0, 1, (n, 1))
y = actificial_data.dot(artificial_weights) + noise
plt.scatter(X, y)



class PolynomialRegression:
    """
    Assumes X: (N, 1)

    """
    def __init__(self, degree):
        self.degree = degree

    def predict(self, w, X, b):
        return X.dot(w) + b

    def loss(self, pred, actual):
        pass

    def fit(self, X, y):
        m, n = X.shape
        poly_x = np.zeros((m, self.degree))

        for i in range(1, self.degree + 1):
            poly_x[:, i - 1] = (X ** i)[:, 0]

        X = poly_x

        self.weight = np.random.uniform(-1, 1, (self.degree, 1))
        self.bias = np.random.uniform(-1, 1, (1, 1))

        for i in range(1000):
            pred = self.predict(self.weight, X, self.bias)

            self.weight = self.weight - 0.01 * (2 * (1.0 / m) * X.T.dot(pred - y))
            self.bias = self.bias - 0.01 * (2 * (1.0 / m) * np.sum(pred - y))


        return self.weight, self.bias



model = PolynomialRegression(degree = 3)
w, b = model.fit(X, y)
print(w, b)

x = np.linspace(-3, 3, num=100)

w_0 = b[0]
w_1 = w[0]
w_2 = w[1]
w_3 = w[2]

# Got this idea from - https://aadhil-imam.medium.com/plotting-polynomial-function-in-python-361230a1e400
fx=[]
for i in range(len(x)):
    fx.append(w_3*x[i]**3 + w_2*x[i]**2 + w_1*x[i] + w_0)

plt.plot(x, fx)

plt.show()

