import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Was racing myself and finished this with no pre-planning, 100% from scratch,
# and not a single google in roughly 1 hour and 40 minutes. 


# Goal:
#Linear(2, 16),
#ReLU(),
#Linear(16, 16),
#ReLU(),
#Linear(16, 1)
#
#SigmoidBinaryCE()



class ReLU:
    def __call__(self, x):
        self.x = x
        self.out = np.maximum(x, 0)
        return self.out

    def backward(self, ingrad):
        return ingrad * (self.x > 0)

class SigmoidBCE: 
    """
    DO NOT SIGMOID BEFORE HAND THIS IS A BYPASS

    """
    def __call__(self, pred, actual):
        self.pred = pred
        self.actual = actual

        m, n = pred.shape

        num_stable = np.maximum(pred, 0) - (pred * actual) + np.log(1.0 + np.exp(-np.abs(pred)))

        self.final_out_shape = pred.shape

        return (1.0 / np.sum(m)) * np.sum(num_stable)
        
    def backward(self, ingrad):
        return ingrad * ((1.0 / (1.0 + np.exp(-self.pred))) - self.actual)



class MoonClassification:
    def fit(self, X, y):
        # Params
        self.X = X
        self.y = y

        self.w0 = np.random.uniform(-1, 1, (2, 16))
        self.b0 = np.random.uniform(-1, 1, (16, 1))

        self.w1 = np.random.uniform(-1, 1, (16, 16))
        self.b1 = np.random.uniform(-1, 1, (16, 1))

        self.w2 = np.random.uniform(-1, 1, (16, 1))
        self.b2 = np.random.uniform(-1, 1, (1, 1))

        # Start Forward
        for i in range(50000):
            self.l0 = X.dot(self.w0) + self.b0.T
            self.l0_act = ReLU()
            self.l0_out = self.l0_act(self.l0)

            self.l1 = self.l0_out.dot(self.w1) + self.b1.T
            self.l1_act = ReLU()
            self.l1_out = self.l1_act(self.l1)

            self.l2 = self.l1_out.dot(self.w2) + self.b2.T

            self.loss = SigmoidBCE()
            self.out = self.loss(self.l2, y)
#            if i % 50 == 0: print(self.out)

            self.backward()

            self.w0 = self.w0 - 0.01 * self.w0_grad
            self.b0 = self.b0 - 0.01 * self.b0_grad

            self.w1 = self.w1 - 0.01 * self.w1_grad
            self.b1 = self.b1 - 0.01 * self.b1_grad

            self.w2 = self.w2 - 0.01 * self.w2_grad
            self.b2 = self.b2 - 0.01 * self.b2_grad

            # SOME BACKWARDS CALL
            # SOME PARAM UPDATE

    def predict(self, X):
        self.l0 = X.dot(self.w0) + self.b0.T
        self.l0_act = ReLU()
        self.l0_out = self.l0_act(self.l0)

        self.l1 = self.l0_out.dot(self.w1) + self.b1.T
        self.l1_act = ReLU()
        self.l1_out = self.l1_act(self.l1)

        self.l2 = self.l1_out.dot(self.w2) + self.b2.T

        self.out = 1.0 / (1.0 + np.exp(-self.l2))

        return self.out

    def backward(self):
        ingrad = np.ones(self.loss.final_out_shape)

        self.loss_grad = self.loss.backward(ingrad) # ingrad 0

        self.b2_grad = np.sum(self.loss_grad, axis = 0, keepdims = True).T

        self.w2_grad = self.l1.T.dot(self.loss_grad) 
        self.l1_grad = self.loss_grad.dot(self.w2.T)
        self.l1_out_grad = self.l1_act.backward(self.l1_grad) # ingrad 1

        self.b1_grad = np.sum(self.l1_out_grad, axis = 0, keepdims = True).T
        self.w1_grad = self.l0.T.dot(self.l1_out_grad) 
        self.l0_grad = self.l1_out_grad.dot(self.w1.T)
        self.l0_out_grad = self.l0_act.backward(self.l0_grad) # ingrad 2

        self.w0_grad = self.X.T.dot(self.l0_out_grad) 
        self.b0_grad = np.sum(self.l0_out_grad, axis = 0, keepdims = True).T






n = 100

X, y = make_moons(n_samples = n)
y = y[:, np.newaxis]
print(y.shape)

model = MoonClassification()

import time

start = time.time()
model.fit(X, y)

print(time.time() - start)





# VISUALIZE

h = 0.01 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Xmesh = np.c_[xx.ravel(), yy.ravel()]

scores = model.predict(Xmesh)

Z = scores.reshape(xx.shape)

fig = plt.figure()

plt.contourf(xx, yy, Z, levels = 1, cmap = plt.cm.ocean, alpha = 0.9)

plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.ocean) 

plt.show()

