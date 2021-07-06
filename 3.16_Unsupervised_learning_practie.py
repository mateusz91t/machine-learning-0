# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 19:17:10 2021

@author: mattt
"""

import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(
    n_samples=100,
    centers=4,
    cluster_std=0.6,  # if bigger then a bigger difference between min and max
    random_state=0  # a seed to get the same values in next shuffles
    )

# X is an array with 2-elements arrays.
X[:5], X[0][1], X[0], len(X)
# If I want get 1st values from each array:
X[:,0][:5]  # it is possible only in np.array(list1), or np.transpose(list1)

plt.scatter(x=X[:,0], y=X[:,1])

# within cluster sum of squares
WCSS = list()


# check how many clusters is the best result
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)


# draw an elbow diagram to get a best clusters' number
plt.plot(range(1, 15), WCSS)
plt.xlabel('Number of K value(Cluster)')
plt.ylabel('WCSS')
plt.grid()
plt.show()  # best value == 4

# learn best 
kmeans = KMeans(n_clusters=4, max_iter=300, random_state=1)
clusters = kmeans.fit_predict(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# prepare data to show()
h = 0.1  # accuracy
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# prepare chart to show()
plt.figure(1, figsize=(15, 7))
plt.clf()

plt.imshow(
    Z,
    interpolation='nearest',
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Pastel1,
    origin='lower')

plt.scatter(x=X[:, 0], y=X[:, 1], c=labels, s=100)

plt.scatter(x=centroids[:, 0], y=centroids[:, 1], s=300, c='red')

plt.ylabel('y'), plt.xlabel('x')
plt.grid()
plt.title('Clustering')
plt.show()
