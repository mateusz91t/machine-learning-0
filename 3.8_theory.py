from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline


iris = pd.read_csv(
    r'D:\Mateusz\nauka\python\Anaconda3\ML\course-files\course-sources\iris.data',
    # \ML\course-files\iris.data',
    header=None,
    names=['petal length', 'petal width',
           'sepal length', 'sepal width', 'species'])

iris.head()
iris.info(memory_usage='deep')
iris.shape
iris.shape[0]
iris.shape[1]
iris[['petal length', 'sepal length', 'species']][40:60]

pl_min, pl_max = iris['petal length'].min() - .5, iris['petal length'].max() + .5
pw_min, pw_max = iris['petal width'].min() - .5, iris['petal width'].max() + .5

colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}


fig, ax = plt.subplots(figsize=(8, 6))

type(plt.subplots(figsize=(8, 6)))
plt.subplots(figsize=(8, 6))
type(fig), type(ax)

igr = iris.groupby(by='species')

for key, group in iris.groupby(by='species'):
    plt.scatter(group['petal length'],
                group['petal width'],
                c=colors[key],
                label=key)

ax.legend()
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(pl_min, pl_max)
plt.ylim(pw_min, pw_max)
ax.set_title('IRIS DATASET CATEGORIZED')

plt.show()


fig, ax = plt.subplots(2, 2, figsize=(10, 6))
plt_position = 1
feature_x = 'petal width'

for feature_y in iris.columns[:4]:
    plt.subplot(2, 2, plt_position)
    for species, color in colors.items():
        plt.scatter(iris.loc[iris['species'] == species, feature_x],
                    iris.loc[iris['species'] == species, feature_y],
                    label=species,
                    alpha=.45,
                    color=color)

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt_position += 1

plt.show()


pd.plotting.scatter_matrix(iris, figsize=(8 , 8),
                           color=iris['species'].apply(lambda x: colors[x]))

import seaborn as sns
sns.set()
sns.pairplot(iris, hue='species')
