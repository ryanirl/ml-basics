#from annoy import AnnoyIndex
#import random
#from SVM import SVM, gaussian
#from KNN import KNN as lame
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_digits
#from KNN import FaissKNeighbors as KNN
#from mlxtend.data import loadlocal_mnist
#import fastdist as dist
#
#X, y = loadlocal_mnist(images_path='./mnist_data/train-images-idx3-ubyte', labels_path='./mnist_data/train-labels-idx1-ubyte')
#
#train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.2, random_state = 10)
#
#f = len(train_data[0])
#t = AnnoyIndex(f, 'euclidean') 
#for i in range(len(train_data)):
#    t.add_item(i, train_data[i])
#
#t.build(3) 
#t.save('test.ann')


