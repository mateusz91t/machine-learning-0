# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:27:15 2021

@author: mattt
"""
# Loading common data related modules
import numpy as np
import pandas as pd
import math

# Loading modelling algorithms
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

# Loading tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Loading visualisation modules
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno

# Configure visualisations
# %matplotlib inline
# plt. SET BACKGROUND

# # Ignore warning messages
# import warnings
# warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# Read data and clear data
diamonds = pd.read_csv('../course-files/course-sources/diamonds.csv')
diamonds.head()
diamonds.drop('Unnamed: 0', axis=1, inplace=True)
diamonds.shape
diamonds.info()
# find nulls
diamonds.isnull().sum()
msno.matrix(diamonds, figsize=(10, 4))  # visualise missing values
diamonds.describe()
diamonds.loc[(diamonds['x'] == 0) | (diamonds['y'] == 0) | (diamonds['z'] == 0)]
len(diamonds.loc[(diamonds['x'] == 0) | (diamonds['y'] == 0) | (diamonds['z'] == 0)])
diamonds[(diamonds[['x', 'y', 'z']] == 0).any(axis=1)].count()
diamonds[(diamonds[['x', 'y', 'z']] == 0).all(axis=1)].count()
diamonds = diamonds[(diamonds[['x', 'y', 'z']] != 0).all(axis=1)]
# always check after execution
diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]


# ----------------------------------------------------------------------------
# Detect dependencies in the data
# Positive or negative corelation:
# if 1st attribute grows up and 2nd attribute grows up too it is a ragnge(0, 1)
# if 1st attribute grows up and 2nd attribute fall down it is a range(-1, 0)
corr = diamonds.corr()
corr
# heatmap: colors instead of numbers of corelations
sns.heatmap(
    data=corr,  # needs rectangular dataset
    annot=True, # shows values in cells
    cbar=True,  # default=True; shows a colorbar
    square=True # means aspect ratio = 'equal', result is a square
    )

# pairplot generates a comparison of each 2 attributes
sns.pairplot(diamonds)

# check distribution of 2 atributes
# we can see there are more diamonds with smaller carat
sns.kdeplot(
    diamonds['carat'],
    shade=True,   # fills area under the line
    color='r',     # changes color and shade
    )

plt.hist(diamonds['carat'])
diamonds['carat'].nunique()
plt.hist(diamonds['carat'], bins=25)

# check a 2 in 1 graph: corelation & distribution
sns.jointplot(
    x='carat', y='price', data=diamonds,
    height=5
    )

# analyze feature by feature, e.g. 'cut' [szlif]
# cut
sns.factorplot(
    x='cut', data=diamonds,
    kind='count',
    aspect=1.5
    )
sns.factorplot(
    x='cut', y='price', data=diamonds,
    kind='box',
    aspect=1.5
    )
# color
sns.factorplot(
    x='color', data=diamonds,
    kind='count',
    aspect=1.5
    )
sns.factorplot(
    x='color', y='price', data=diamonds,
    kind='violin',
    aspect=1.5
    )
# clarity
sns.factorplot(
    x='clarity', data=diamonds,
    kind='count',
    aspect=1.5
    )
sns.factorplot(
    x='clarity', y='price', data=diamonds,
    kind='violin',
    aspect=1.5
    )
clarity_labels = diamonds['clarity'].unique()
clarity_labels
clarity_sizes = diamonds.clarity.value_counts()
clarity_sizes
colors = ['#006400', '#E40E00', '#A00994', '#613205', '#FFED0D',
          '#16F5A7', '#ff9999', '#66b3ff']
clarity_explode = tuple([0.1 for i in range(8)])
clarity_explode, len(clarity_explode)

plt.pie(
        clarity_sizes,
        explode=clarity_explode,
        labels=clarity_labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,  # small shadow under pieces of pie
        startangle=0  # rotation of a pie
        )

plt.axis('equal')
plt.title('Percentage of clarity categories')
plt.plot()
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.show()


sns.boxplot(x='clarity', y='price', data=diamonds)

plt.hist(
    'depth', data=diamonds,
    bins=25
    )

sns.jointplot(
    x='depth', y='price', data=diamonds,
    height=5
    )

sns.kdeplot(
    diamonds['table'],
    shade=True,
    color='orange'
    )

sns.jointplot(
    x='table', y='price', data=diamonds,
    height=5
    )


# ------------------------------------------------------------------------
# Preparing data to Machine Learning

# feature engineering - analyzing separately xyz doesn't make sense
sns.kdeplot(diamonds['x'], shade=True, color='r')
sns.kdeplot(diamonds['y'], shade=True, color='g')
sns.kdeplot(diamonds['z'], shade=True, color='b')
plt.xlim(2, 10)  # limit of x axes
diamonds['volume'] = diamonds['x'] * diamonds['y'] * diamonds['z']
diamonds.head()

plt.figure(figsize=(5, 5))
plt.hist(x=diamonds['volume'], bins=30, color='g')
plt.xlabel('Volume in mm^3')
plt.ylabel('Frequency')
plt.title('Distribution of diamond\'s volume')
plt.xlim(0, 1000)
plt.ylim(0, 50000)

sns.jointplot(x='volume', y='price', data=diamonds, height=5)
diamonds.drop(['x', 'y', 'z'], axis=1, inplace=True)
diamonds.head()

# One hot encoding
