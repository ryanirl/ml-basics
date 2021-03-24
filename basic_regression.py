import numpy as np
import matplotlib.pyplot as plt

n = 200

X = np.random.rand(n)[np.newaxis]

# Cool trick I learned from Percy Liang during one of his lectures for CS221.
# It generates fake Y values and if the program works then w and b will converge
# to whatever you set these values too.
artificial_weights = np.array([[1]])
artificial_bias = np.array([[3]])

noise = np.random.rand(n)
Y = np.dot(artificial_weights.T, X) + artificial_bias + (noise / 5.0)
plt.scatter(X, Y)

def hypothesis(w, x, b):
    return np.dot(w.T, x) + b

def train(w, b, x, y):
#    print("x: {}".format(np.shape(x)))
#    print("y: {}".format(np.shape(y)))
#    print("w: {}".format(np.shape(w)))
#    print("b: {}".format(np.shape(b)))

    for i in range(10000):
        h = hypothesis(w, x, b)

        w = w - ( 0.01 * (2.0 * np.dot(h - y, x.T) / 200.0))
        b = b - ( 0.01 * (2.0 * np.sum(h - y)) / 200.0 )
        print("w: {}, b: {}".format(w, b))
        
    return w, b

    


w_init = np.array([[0]])
b_init = np.array([[0]])
w, b = train(w_init, b_init, X, Y)
print(w)
print(b)

#x = np.linspace(-20, 20, 20)
#hyperplane = w[0] * x + b
#
#plt.plot(x, hyperplane, '-', color="blue")
#
#plt.show()
