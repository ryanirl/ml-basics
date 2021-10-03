import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KMeans:
    def __init__(self, k_clusters = 2):
        self.k_clusters = k_clusters

    def fit(self, X):
        self.X = X

        # Randomly selects 'self.k_clusters' samples from out data to initialize the points
        # as. This is done with no replacement as to avoid duplicates.
        self.centers = X[np.random.choice(X.shape[0], self.k_clusters, replace=False)]

        # Expands 'X' so we can broadcast it with 'self.centers'. 
        X = np.tile(X, self.k_clusters).reshape(200, self.k_clusters, self.X.shape[1])

        # Initialize starting values for while loop
        new_cluster = np.array([1.0])
        old_cluster = np.array([0.5]) 
        
        # While there are updates to new_cluster
        while not np.array_equal(new_cluster, old_cluster):
            old_cluster = new_cluster

            new_cluster = np.argmin(np.linalg.norm(self.centers - X, axis = -1), axis = 1)

            # This can be sped up by not including the loop.
            # I just don't know how.
            for i in range(self.k_clusters):
                index = np.where(new_cluster == i) 
                self.centers[i] = np.mean(self.X[index], axis = 0)


        return self.centers

    def plot(self, actual):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')

        colors = ["red", "green", "blue", "yellow", "black"]
        for i in range(self.k_clusters):
            for j in range(len(self.X[:, 0])):
                if (actual[j] == i):
                    ax.scatter3D(self.X[j][0], self.X[j][1], self.X[j][2], color = colors[i])

        for i in range(self.k_clusters):
            ax.scatter3D(self.centers[i][0], self.centers[i][1], self.centers[i][2], color = "black", s = 200)


        plt.show()


X, Y = make_blobs(n_samples = 200, n_features = 3, centers = 4, cluster_std = 0.8, random_state = 6)

model = KMeans(4)
model.fit(X)
model.plot(actual = Y)







