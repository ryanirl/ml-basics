import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import numpy as np
from mlxtend.plotting import plot_decision_regions

# This is the very easy way to impliment SVM with Gaussian Kernal and I plan on doing 
# a "from scratch" implimentation along with math behind some of it.


# Create Dataset
# Copied this data set from this acticle: 
# https://towardsdatascience.com/how-to-learn-non-linear-separable-dataset-with-support-vector-machines-a7da21c6d987
np.random.seed(0)

X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = np.where(y, 1, -1)

# If you want circular data points instead
#X, y = make_circles()

model = SVC(kernel='rbf')
clf = model.fit(X, y)

# mlxtend library for plotting decision regions is extremely convienient
plot_decision_regions(X=X, y=y, clf=clf, legend=2)
plt.show()

