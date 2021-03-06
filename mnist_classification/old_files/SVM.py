import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from sklearn.datasets import make_circles
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from scipy.spatial.distance import pdist, squareform
import scipy

# HUGE OVERFITTING PROBLEM
# SOLUTION: RAISE SIGMA VALUE 

def gaussian(x1, x2, sigma = 0.1, axis = 1):
    return np.exp(-(np.linalg.norm(x1 - x2, axis=axis) ** 2) / (2.0 * (sigma ** 2.0)))

# Takes roughly 24 minutes to compute linear SVM with 60,000 images OUCH
def linear(x1, x2):
    return np.dot(x1, x2.T)

# Much of the design is based off of Aladdin Persson's implimentation, his video on SVM with CVXOPT was
# instrumental in my understanding of the CVXOPT library, so huge shoutout to him. The link to his video is:
# https://www.youtube.com/watch?v=gBTtR0bs-1k&list=PLhhyoLH6IjfxpLWyOgBt1sBzIapdRKZmj&index=7
# ANY CODE I DIRECTLY COPIED WILL BE INITIALED "AP"

class SVM:
    '''
    Much of the function names are bassed on names from sklearn library. The visualization
    library I use (mlxtend) was built for the sklearn library so I mimicked the function
    names in order to get it to work with my implimentation.

    mlxtend requires SVM to be an class with functions predict and fit
    
    '''

    def __init__(self, kernel = gaussian, sigma = 0.6, C = 1): 
        self.kernel = kernel
        self.sigma = sigma
        self.C = C


    def fit(self, X, y):
        m, n = np.shape(X)
        self.y = y
        self.X = X
        self.gram_matrix = np.zeros((m, m))


        # AP - This is much quicker than my implimentation as it
        # gets rid of an extra for loop. 
        for i in range(m):
            print(i)
            self.gram_matrix[i, :] = linear(X[i, np.newaxis], self.X)

#        # Kernel K<X, X>
#        for i in range(m):
#            print(i)
#            for j in range(m):
#                self.gram_matrix[i][j] = linear(X[i], X[j])


        # The following article was a big help in understanding the conversion from dual form
        # to CVXOPT required form and implimenting CVXOPT:
        # https://xavierbourretsicotte.github.io/SVM_implementation.html

        P = cvxopt.matrix(np.outer(y, y) * self.gram_matrix)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((-np.identity(m), np.identity(m))))
        h = cvxopt.matrix(np.vstack((np.zeros((m, 1)), np.ones((m, 1)) * self.C)))
        A = cvxopt.matrix(y, (1, m), "d")
        b = cvxopt.matrix(np.array([0]), (1, 1), 'd')

        optimal = cvxopt.solvers.qp(P, q, G, h, A, b)
        cvxopt.solvers.options['show_progress'] = True

        self.alphas = np.array(optimal["x"])
      

    def predict(self, X):
        '''
        I will post notes on how we derive a prediction given alphas
        '''

        prediction = np.zeros(len(X))

        S = ((self.alphas > 1e-4)).flatten()

        self.bias = np.mean(self.y[S] - self.alphas[S] * self.y[S] * self.gram_matrix[S, S][:, np.newaxis]) #AP

        # Kernel <X_i, self.X>
        for i in range(len(X)):
            prediction[i] = np.sum(self.alphas[S] * self.y[S] * linear(X[i], self.X[S], self.sigma, axis=1)[:, np.newaxis]) #AP
            
        return np.sign(prediction + self.bias) #AP










