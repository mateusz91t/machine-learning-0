# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 11:03:12 2021

@author: mattt
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

auto = pd.read_csv(
    './../course-files/course-sources/auto-mpg.csv')
auto.head()
auto.info()
auto.shape

X = auto.loc[
    :,
    list(filter(lambda x: x not in ['mpg', 'car name'], auto.columns))
]
X.info()

# clean data
# try stay NaN
X['horsepower'].replace('?', 0, inplace=True)
y = auto['mpg']

# remove NaN
X = X[auto['horsepower'].str.isdigit()]
y = auto[auto['horsepower'].str.isdigit()]['mpg']

# convert values to number
X['horsepower'] = X['horsepower'].astype('int')

# or drop a NaN column
X = X.drop('horsepower', axis='columns')
y = auto['mpg'][X.index]

y.head()

lr = LinearRegression()
lr.fit(X, y)
lr.score(X, y)


# some cars that need be evaluated [with horsepower]
my_car1 = [4, 160, 190, 2000, 12, 90, 1]
my_car2 = [4, 200, 260, 2500, 15, 83, 1]
cars = [my_car1, my_car2]

# some cars that need be evaluated [without horsepower]
my_car1 = [4, 160, 190, 12, 90, 1]
my_car2 = [4, 200, 260, 15, 83, 1]
cars = [my_car1, my_car2]

lr.predict(cars)
