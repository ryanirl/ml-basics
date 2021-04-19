# ML-BASICS

### WORK IN PROGRESS

Implimentation of some basic Machine Learning Models with Python. Topics including Support Vector Machines (SVM), Logistic Regression, K-Means Clustering, and more. Please contact me with any questions, bugs, or typos.

<br />

<p align="center">
 <img src="./img/new_preview.png" width="95%">
</p>

<br />

**Currently Working On:** Basic Neural Networks 

*PROBLEMS:* Hit a roadblock where I don't have the proper prerequisites in probability to understand bayesian "stuff" and so I have been spending some time taking MIT OCW's 18.600 to better prepare myself. This will slow down the progress in this repo.
 
<br />
 
---

## Road Map:

**BOLDS** are things I have begun.

### Supervised Learning
1. Classification
    - **Logistic Regression** & Multinomial Logistic Regression
    - **Support Vector Machines (SVM)**
    - Naive Bayes
    - **K-Nearest-Neighbors (KNN)**
    - Decision Trees & Random Forest
    
2. Regression
    - **Linear Regression**
    - **Polynomial Regression**
    - Lasso Regression (L1)
    - Ridge Regression (L2)
    - **Partial Lease Squares (PLS)**
    - Principle Component (PCR)

3. Neural Networks
    - CNN
    - RNN
    - Transformer Networks
    - Generative Adversarial Nets (GAN)

4. Boosting
    - XGBoost
    - AdaBoost
    - Gradient Tree Boosting
    

### Unsupervised Learning
1. Clustering
    - **K-Means**
    - Mean Shift
    - Expectation Maximization (EM) with Gaussian Mixture Models (GMM)

2. Dimensionality Reduction
    - Principle Component Analysis (PCA)
    
    
### Analysis of Model
1. Classification
    - **Accuracy**
    - Precision
    - Recall
    - f1
    - **Confusion Matrix**
    - Mean Average Precision

2. Regression
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - R squared

3. Bias and Variance of Models

## Basic Projects
1. mnist Classification 
2. Car Price Classification 
    - Data taken from my CraigslistScraper repo :)

3. Cart-Pole Balancing System
4. ??? NEED IDEAS ??? 


<br />

---


## MNIST Classification with KNN

The MNIST dataset is a collection of handwritten digits than we can then use to train some classification model that can make predictions given new handwritten digits. This is a pretty intuitive example:

<br />

<p align="center">
 <img src="./mnist_classification/img/mnist_classification_example.png" width="75%">
</p>

<br />

Using the K-Nearest-Neighbor model with the Annoy library to optimize via approximations and avoid calculating the computationally expensive L2 norm for all 60,000 datapoints in the mnist dataset I was able to create a model that can train on the entire MNIST dataset in under 20 seconds on a Macbook Pro with an average 95% accuracy. Plotting its confusion matrix gives:

<br />

<p align="center">
 <img src="./mnist_classification/img/mnist_confusion_matrix_knn.png" width="50%">
</p>

<br />

Find the code and more detail in my jupyter notebook: https://github.com/ryanirl/ml-basics/tree/main/mnist_classification/KNN_mnist.ipynb

<br />




