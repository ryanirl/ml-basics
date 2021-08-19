import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


### --- Hyperparameters --- ###

n = 100 
eta = 0.01
epochs = 5000


### --- Gather Dataset --- ###

X, y = make_blobs(n_samples = n, centers = 2)

y = y[:, np.newaxis]


### --- Build Model --- ###

def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegression:
    def predict(self, X, w, b):
        return sigmoid(X.dot(w) + b)

    def loss(self, pred, y):
        BCE = (y * np.log(pred + 1e-6) + (1 - y) * np.log(1 - pred + 1e-6)) 
        return -np.sum(BCE) * (1.0 / self.m)

    def fit(self, X, y):
        self.m, self.n = X.shape

        self.weights = np.random.uniform(-1, 1, (self.n, 1))
        self.bias = np.random.uniform(-1, 1, (1, 1))

        for i in range(epochs):
            predicted = self.predict(X, self.weights, self.bias)

            if i % 100 == 0: print("Loss on step {} is: {}".format(i, self.loss(predicted, y)))

            self.weights = self.weights - eta * X.T.dot(predicted - y)
            self.bias = self.bias - eta * np.sum(predicted - y)



### --- Instantiate Model --- ###

model = LogisticRegression()
model.fit(X, y)

weight0, weight1 = model.weights
bias = model.bias[0][0]


### --- Plot --- ###

for i in range(n):
    if (y[i] == 1):
        plt.scatter(X[:, 0][i], X[:, 1][i], color="green") 
    else:
        plt.scatter(X[:, 0][i], X[:, 1][i], color="blue")

        
x = np.linspace(-5, 5, 5)

hyperplane = (-(weight0 / weight1) * x) - (bias / weight1)

plt.suptitle("Logistic Regression")

plt.plot(x, hyperplane, '-', color = "red")

plt.show()

