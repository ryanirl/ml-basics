import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from sklearn.datasets import make_circles
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC


# HUGE OVERFITTING PROBLEM
# SOLUTION: RAISE SIGMA VALUE 

def gaussian(x1, x2, sigma = 0.1, axis = 0):
    return np.exp(-(np.linalg.norm(x1 - x2, axis=axis) ** 2) / (2.0 * (sigma ** 2.0)))

def polynomial(x1, x2, p = 1, sigma = 0.1, axis = 0):
    return (1.0 + (np.dot(x2, x1))) ** p

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
                self.gram_matrix[i][j] = gaussian(X[i], X[j], self.sigma, axis=0) 

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
            prediction[i] = np.sum(self.alphas[S] * self.y[S] * gaussian(X[i], self.X[S], self.sigma, axis=1)[:, np.newaxis]) #AP
            
        return np.sign(prediction + self.bias) #AP



# --- Circle Test --- # 
#X, y = make_circles(random_state = 1)
# ------------------- # 



if __name__ == "__main__":
    # --- Overfit Test --- # 
    np.random.seed(0)

    X = np.random.randn(200, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    y = np.where(y, 1, -1)
    # -------------------- #



    y[y == 0] = -1
    y_int_32 = y
    y = y.reshape(-1,1) * 1.

    # Multiple Plots 
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle("SVM w/ Gaussian Kerel")
    ax0.set_title("FROM SCRATCH")
    ax1.set_title("SKLEARN")

    # My Implimentation
    model = SVM()
    model.fit(X, y)
    plot_decision_regions(X, y_int_32, clf=model, legend=2, ax=ax0)

    # sklearn Implimentation
    sklearn_model = SVC(kernel='rbf')
    sklearn_model.fit(X, y_int_32)
    plot_decision_regions(X, y_int_32, clf=sklearn_model, legend=2, ax=ax1)

    # SHOW PLOT
    plt.show()






