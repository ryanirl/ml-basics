from SVM import SVM, gaussian
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X, y = loadlocal_mnist(images_path='./mnist_data/train-images-idx3-ubyte', labels_path='./mnist_data/train-labels-idx1-ubyte')

#print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
#print('\n1st row', X[0])
#new = np.zeros((60000, 28, 28))
#
#for i in range(60000):
#    new[i] = X[i].reshape(28, 28)

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title("True Label: {}".format(y[idx]))
    plt.show()

#model = SV(kernel='rbf') clf = model.fit(X, y)

test = SVM()
test.fit(X, y)












