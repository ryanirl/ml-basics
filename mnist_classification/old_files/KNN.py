import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import faiss
from annoy import AnnoyIndex


# About 15-20 seconds to work through
# the whole original MNIST dataset.
class AnnoyKNN:
    def __init__(self, K = 3):
        self.K = K

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.m, self.n = np.shape(X_train)

    def predict(self, X_test):
        test_m, test_n = np.shape(X_test)

        # Initializes Annoy index / matrix
        matrix = AnnoyIndex(self.n, 'euclidean')

        # Constructing Annoy matrix
        for i in range(self.m):
            matrix.add_item(i, self.X_train[i])

        # Builds Annoy matrix
        matrix.build(self.K)

        # Building prediction matrix
        predictions = np.zeros(test_m)


        for i in range(test_m):
            # Compiles k nearest neighbors for each number / data point
            # in out testing data and returns their index
            index = matrix.get_nns_by_vector(X_test[i], self.K)

            # Given our indexes above, this is piece of code "votes"
            # and returns the majority nearest neighbor
            predictions[i] = np.argmax(np.bincount(self.y_train[index]))

        return predictions


# 100% copy and pasted from here:
# https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
# This acticle introduced me to a whole field that works on calculating nearest neighbors which
# is how I found the Annoy python library in which I did my optimized implimentation on
class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        print(indices)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


# Accury and Speed not the best
# But more intuitive than either
# FaissKNeighbors or AnnoyKNN
class KNN:
    def __init__(self, K):
        self.K = K

    def _distance(self, x, z):
        return np.linalg.norm(x - z, axis = 1)

    def fit(self, X_train, y_train):
        """
        Because all computation is done during predict, nothing
        is needed here.

        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        self.X_test = X_test

        m, n = np.shape(X_test)
        predictions = np.zeros(m)

        for i in range(m):
            print(m)
            distance = self._distance(self.X_test[i], self.X_train)

            index = np.argsort(distance, axis = 0, kind="heapsort")[: self.K]

            predictions[i] = np.argmax(np.bincount(self.y_train[index]))

        return predictions



def main():
    # Heavily processed mnist data | SIZE (1797, 64)
    mnist = load_digits()

    # Splitting into training and testing
    train_data, test_data, train_labels, test_labels = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state = 1)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    model = AnnoyKNN(K = 3)
    model.fit(train_data, train_labels)
    predict1 = model.predict(test_data)


    predict1[predict1 != test_labels] = 0
    predict1[predict1 == test_labels] = 1

    right = np.sum(predict1)

    accuracy = float(right) / len(predict1)

    print("accuracy: {}".format(accuracy))


if __name__ == "__main__":
    main()





