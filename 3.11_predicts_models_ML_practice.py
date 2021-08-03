# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 21:12:22 2021

@author: mattt
"""

# Linear Regression: an algorithm that allows to predict continuous values [wartości ciągłe]
# Continuous values: real numbers [liczby rzeczywiste], e.g. temperature, speed
# Discrete values[wartoci dyskretne]: categories, e.g. Male/Female, Country

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


iris = pd.read_csv(
    r'D:\Mateusz\nauka\python\Anaconda3\ML\course-files\course-sources\iris.data',
    # \ML\course-files\iris.data',
    header=None,
    names=['petal length', 'petal width',
           'sepal length', 'sepal width', 'species'])

# split data into features (X) and labels (y)
X = iris.iloc[:, :4]  # iloc by column id
y = iris.loc[:, 'species']  # loc by column name

iris_types = iris['species'].unique()
type(iris_types)
categories = dict()
for i in range(len(iris_types)):
    categories[iris_types[i]] = i + 1

iris_types
categories

X.head()
y.head()

y = y.apply(lambda x: categories[x])
y.head()


# model in 3 lines...
lr = LinearRegression()
lr.fit(X, y)
lr.score(X, y)


# some data examples that need to be evaluated
iris_1 = [5, 3.5, 1.4, 0.2]
iris_2 = [6.4, 3, 4.5, 1]
iris_3 = [6, 3, 5, 2]
other = [1, 2, 3, 4]
flowers = [iris_1, iris_2, iris_3, other]


# running a prediction in the model
species_predict = lr.predict(flowers)
print(species_predict)


# replacing continous values to discrete values
for f, s in zip(flowers, species_predict):
    result = round(s)
    if result in categories.values():
        print(f'flower {f} is {iris_types[result - 1]}')
    else:
        print(f'flower {f} is UNKNOWN')
