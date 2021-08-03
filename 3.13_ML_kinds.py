# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:51:05 2021

@author: mattt
"""

# One row in DB == sample in ML [próbka]
# Columns in ML are called properties [właściwości]
# or attribute, feature [cecha]
# A main column to predict is called target or label [etykieta]


# ML types:
#   supervised learning [uczenie nadzorowane]:
#       initial data should be described (known)
#       good prepared data with out values to predict this values
#       * categorization algorithms
#   unsupervised learnig:
#       if there is too much data
#       * categories
#       * clustering
#       * a dimension reduction [redukcja wymiarów] if there are not important
#           attributes
#       * anomaly detection [wykrywanie anomalii]
#
#   reinforcement learning:
#       reward and punishment
#       if an algorithm can get information
#           about its good or bad decision
#       for classification problem
#       Regression algorithms:
#           * Linear Regression
#           * k-Neighbours
#           * Random Forest
#           * Neural networks

# We can use regression algorithm from reinforcement learning
# or clustering from unsupervised learning
# to find categories and later
# we can use categorization from supervised learning to predict
# a category for next object


# 3.15 Unsupervised learning practice

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv(
    '../course-files/course-sources/Airbnb listings in Ottawa (May 2016).csv')

df.head()
df.shape


coordinates = df.loc[:, ['longitude', 'latitude']]
plt.scatter(df.loc[:, 'longitude'], df.loc[:, 'latitude'])
coordinates

# within cluster sum of squares
WCSS = list()

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(coordinates)
    WCSS.append(kmeans.inertia_)


# show an elbow chart [łokieć - ze względu na kształt linii]
plt.plot(range(1,15), WCSS)
plt.xlabel('Number of K Value(Cluster)')
plt.ylabel('WCSS')
plt.grid()
plt.show()  # best value is 5

# learn best
kmeans = KMeans(n_clusters=5, max_iter=300, random_state=1)
clusters = kmeans.fit_predict(coordinates)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# prepare data to show()
h = 0.001  # accuracy
# borders' variables of a chart
x_min, x_max = coordinates['longitude'].min(), coordinates['longitude'].max()
y_min, y_max = coordinates['latitude'].min(), coordinates['latitude'].max()
# a matrix with all points in a chart
x_arange, y_arange = np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(x_arange, y_arange)
# a predictction result of xx and yy
xx_ravel, yy_ravel = xx.ravel(), yy.ravel()
c_ravelled = np.c_[xx_ravel, yy_ravel]
Z = kmeans.predict(c_ravelled)


# now we are ready to draw a chart
plt.figure(1, figsize=(10, 4))
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(
    Z,
    interpolation='nearest',
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap = plt.cm.Pastel1,
    origin='lower')

plt.scatter(x=coordinates['longitude'],
            y=coordinates['latitude'],
            c=labels,
            s=100)

plt.scatter(
    x=centroids[:, 0],
    y=centroids[:, 1],
    s=300,
    c='red')

plt.ylabel('Long(y)'), plt.xlabel('Lat(x)')
plt.grid(), plt.title('Clustering')
plt.show()
