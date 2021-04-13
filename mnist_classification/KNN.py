import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import faiss


# Found here: https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
# I mean... wow, this thing eats up the whole mnist dataset in like 15 seconds. 
# I will be working out my own twist on this once I do some studying on the faiss library
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
            distance = self._distance(self.X_test[i], self.X_train)

            index = np.argsort(distance, axis = 0, kind="heapsort")[: self.K]
            print(index)

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

    model = KNN(K = 3)
    model.fit(train_data, train_labels)
    predict1 = model.predict(test_data)


    predict1[predict1 != test_labels] = 0
    predict1[predict1 == test_labels] = 1

    right = np.sum(predict1)

    accuracy = float(right) / len(predict1)

    print("accuracy: {}".format(accuracy))


if __name__ == "__main__":
    main()





