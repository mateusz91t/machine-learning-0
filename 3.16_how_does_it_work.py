from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# how to get a desired dimension [pożądany wymiar]
l1 = [[1, 2],
      [3, 4],
      [5, 6]]
type(l1)

l1[1]
l1[:2]
# l1[:, 1]~ # it doesn't work!

# by pandas DF
df1 = pd.DataFrame(l1)
# df1[:, 1]  # is doesn't work
df1[1]  # it works
df1.iloc[:, 1]  # it works
df1.loc[:, 1]  # same values beacause cols names and id are the same

all(df1[1] == df1.iloc[:, 1])

# pandas by transpose
df1.transpose().loc[1]


# by numpy
a1 = np.array(l1)
a1
a1[:, 1]
a1.transpose()[1]


# random (if want) data to an analysis
data1 = make_blobs(
    # n_samples=100,
    # n_features=2,  # what is it?
    centers=4,      # default=None
    # if bigger center box, we need bigger cluster_std
    cluster_std=5,  # default=1, distances between clusters
    center_box=(10, 100),  # default=(-10.0, 10.0); min and max
    random_state=2  # default=None, if we want get the same result, set it
    )
X, y = data1
X
y

plt.scatter(x=X[:, 0], y=X[:, 1])


# clustering: check best number of clusters
# within cluster sum of squares -
# Sum of squared distances of samples to their closest cluster center
# [Suma kwadratów odległoci próbek do najbliższego srodka klastra]
wcss = list()

for i in range(1, 15):
    kmeans = KMeans(n_clusters=i)   # generates a k-Means model
    kmeans.fit(X)           # finds centroids
    wcss.append(kmeans.inertia_)    # save an avg error to a list

wcss

# draw an elbow diagram from wcss
plt.plot(wcss)
plt.plot(wcss, range(1, 15))
plt.plot(range(1, 15), wcss)
# best == 4 or 5


# clustering: learn best
kmeans = KMeans(n_clusters=5,
                )
kmeans
# Compute cluster centers and predict cluster index for each sample.
# Wchich point belongs [należy] to which cluster.
clusters = kmeans.fit_predict(X)
clusters
# Labels for each point:
# it contains info about which sample belongs to which cluster
labels = kmeans.labels_
labels
all(clusters == labels)  # wtf ?? the same?
# Coordinates of cluster centers
centroids = kmeans.cluster_centers_
centroids


# prepare clusters to show
h = 0.1  # accuracy
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_min, x_max
y_min, y_max
# arange: generate points from start to stop with h-step
arange_x = np.arange(x_min, x_max, h)
arange_y = np.arange(y_min, y_max, h)
arange_x
arange_y
len(arange_x), len(arange_y)
plt.plot(arange_x)
plt.plot(arange_y)
# meshgrid it is a table with values which grow up in x or y axis
meshgrid = np.meshgrid(arange_x, arange_y)
type(meshgrid)
len(meshgrid)
meshgrid[1]
len(meshgrid[0])
len(meshgrid[1])
xv, yv = meshgrid
len(xv)
xv
# ravel(): sticks each column to end of first column. Returns 1 long column
xv_raveled, yv_raveled = xv.ravel(), yv.ravel()
xv_raveled
len(xv[:, 1]), len(xv[1]), len(xv_raveled), len(xv[:, 1]) * len(xv[1])
plt.plot(xv_raveled)
# c_ : Translates slice objects to concatenation along the second axis.
# returns 2 columns of all variants of mix xv_raveled and yv_raveled
c_np = np.c_[xv_raveled, yv_raveled]
c_np
len(c_np)


# what is it?
Z = kmeans.predict(c_np)
xv.shape
len(Z)
Z
Z = Z.reshape(xv.shape)
len(Z)
Z


# draw a final chart
plt.figure(1, figsize=(15, 7))
plt.clf()

# 1st backgroud layer
# prepare a x-y chart with divide area
plt.imshow(
    Z,                      # dataset
    cmap=plt.cm.Pastel1,    # color
    aspect=0.60,            # image's shape -> square or rectangle [prostokąt]
    interpolation='nearest',  # curve from a point to point ?
    origin='lower',         # y values group up or fall down
    extent=(xv.min(), xv.max(), yv.min(), yv.max()) # min & max values of x & y
    )

# 2nd backgroud layer
# draw points
plt.scatter(
    X[:, 0],   # x
    X[:, 1],   # y
    s=40,      # makes points bigger/smaller
    c=labels   # divides points into groups, which point belongs to which group
    )

# 3rd background layer
# draw centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=60,
    c='red'
    )

# 4th background layer
plt.ylabel('y'), plt.xlabel('x')
plt.grid()
plt.title('clustering understanding')
plt.show()