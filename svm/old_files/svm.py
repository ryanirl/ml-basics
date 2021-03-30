import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from sklearn.datasets import make_circles
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC


# HUGE OVERFITTING PROBLEM
# SOLUTION: RAISE SIGMA VALUE 

# NEED MORE GENERAL GAUSSIAN
def gaussian(x1, x2, sigma = 0.1):
    return np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2.0 * (sigma ** 2.0)))

def gaussiantest(x1, x2, sigma = 0.1):
    return np.exp(-(np.linalg.norm(x1 - x2, axis=1) ** 2) / (2.0 * (sigma ** 2.0)))

# Much of the design is based off of Aladdin Persson's implimentation, his video on SVM with CVXOPT was
# instrumental in my understanding of the CVXOPT library, so huge shoutout to him. The link to his video is:
# https://www.youtube.com/watch?v=gBTtR0bs-1k&list=PLhhyoLH6IjfxpLWyOgBt1sBzIapdRKZmj&index=7

class SVM:
    '''
    Much of the function names are bassed on names from sklearn library. The visualization
    library I use (mlxtend) was built for the sklearn library so I mimicked the function
    names in order to get it to work with my implimentation.

    mlxtend requires SVM to be an class with functions predict and fit
    
    '''

    def __init__(self, kernel = gaussian, sigma = 0.6, C = 1): 
        self.kernal = kernel
        self.sigma = sigma
        self.C = C


    def fit(self, x, y):
        m, n = np.shape(x)
        self.y = y
        self.X = X
        self.gram_matrix = np.zeros((m, m))

        # Kernel K<X, X>
        for i in range(m):
            for j in range(m):
                self.gram_matrix[i][j] = gaussian(X[i], X[j], self.sigma) 

        P = cvxopt.matrix(np.outer(y, y) * self.gram_matrix)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((-np.identity(m), np.identity(m))))
        h = cvxopt.matrix(np.vstack((np.zeros((m, 1)), np.ones((m, 1)) * self.C)))
        A = cvxopt.matrix(y, (1, m), "d")
        b = cvxopt.matrix(np.array([0]), (1, 1), 'd')

        optimal = cvxopt.solvers.qp(P, q, G, h, A, b)
        cvxopt.solvers.options['show_progress'] = True

#        # The following article was a big help in understanding the conversion from dual form
#        # to CVXOPT required form and implimenting CVXOPT:
#        # https://xavierbourretsicotte.github.io/SVM_implementation.html
#        # The following code for determining x, S, and b are copied from this article

        self.alphas = np.array(optimal["x"])
      

    def predict(self, X):
        '''
        I will post notes on how we derive a prediction given alphas
        
        '''

        prediction = np.zeros(len(X))

        S = ((self.alphas > 1e-4)).flatten()

        self.bias = np.mean(self.y[S] - self.alphas[S] * self.y[S] * self.gram_matrix[S, S][:, np.newaxis])

        for i in range(len(X)):
            prediction[i] = np.sum(self.alphas[S] * self.y[S] * gaussiantest(X[i], self.X[S], self.sigma)[:, np.newaxis])

        return np.sign(prediction + self.bias)



         



# --- Circle Test --- # 
#X, y = make_circles(random_state = 1)
# ------------------- # 



# --- Overfit Test --- # 
np.random.seed(0)

X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = np.where(y, 1, -1)
# -------------------- #



y[y == 0] = -1
y_int_32 = y
y = y.reshape(-1,1) * 1.

# My Implimentation
model = SVM()
model.fit(X, y)
plot_decision_regions(X, y_int_32, clf=model, legend=2)
plt.show()


## sklearn Implimentation
#sklearn_model = SVC(kernel='rbf')
#clf = sklearn_model.fit(X, y_int_32)




