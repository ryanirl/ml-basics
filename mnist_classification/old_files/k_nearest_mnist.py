from SVM import SVM, gaussian
from KNN import KNN as lame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from KNN import AnnoyKNN as KNN
from mlxtend.data import loadlocal_mnist
import fastdist as dist
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix

X, y = loadlocal_mnist(images_path='./mnist_data/train-images-idx3-ubyte', labels_path='./mnist_data/train-labels-idx1-ubyte')

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.2)


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


# If you would like to plot an image
def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title("True Label: {}".format(y[idx]))
    plt.show()


model = KNN(K = 10)
model.fit(train_data, train_labels)

predict1 = model.predict(test_data)


y_target = test_labels 
y_predicted = predict1 

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted, 
                      binary=False)


fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()

predict1[predict1 != test_labels] = 0
predict1[predict1 == test_labels] = 1

right = np.sum(predict1)

accuracy = float(right) / len(predict1) 

print("accuracy: {}".format(accuracy))











