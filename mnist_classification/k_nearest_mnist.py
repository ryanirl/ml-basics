from SVM import SVM, gaussian
from KNN import KNN as lame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from KNN import FaissKNeighbors as KNN

X, y = loadlocal_mnist(images_path='./mnist_data/train-images-idx3-ubyte', labels_path='./mnist_data/train-labels-idx1-ubyte')

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.2)
#
test_data = test_data[: 10]
test_labels = test_labels[: 10]

### SECOND DATA SET ## 
#
## Heavily processed mnist data | SIZE (1797, 64)
#mnist = load_digits()
#
## Splitting into training and testing
#train_data, test_data, train_labels, test_labels = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state = 1)
#
#
######################


def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title("True Label: {}".format(y[idx]))
    plt.show()

model = KNN(k = 10)
model.fit(train_data, train_labels)

predict1 = model.predict(test_data)


predict1[predict1 != test_labels] = 0
predict1[predict1 == test_labels] = 1

right = np.sum(predict1)

accuracy = float(right) / len(predict1) 

print("accuracy: {}".format(accuracy))



model = lame(K = 10)
model.fit(train_data, train_labels)

predict1 = model.predict(test_data)


predict1[predict1 != test_labels] = 0
predict1[predict1 == test_labels] = 1

right = np.sum(predict1)

accuracy = float(right) / len(predict1) 

print("accuracy: {}".format(accuracy))










