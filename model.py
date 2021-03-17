# -*- coding: utf-8 -*-
"""
Created on 17-03-2021

@author: Casper Kanaar 

%DESCRIPTION
"""
# =============================================================================
# Importing relevant modules 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
 
# =============================================================================
# Importing and formatting the data.
data        = pd.read_csv("data/data.csv",sep = ";")

y = data.pop("Prijs")                                   # [-]. Output vector 
X = data                                                # [-]. Design matrix

train       = 0.7                                       # [-]. Train ratio
test        = 1 - train                                 # [-]. Test ratio 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test)

assert np.isclose(len(X_train)+len(X_test),len(data)), "Implementation error: Train and Test data not properly separated."

# =============================================================================
# Building the classification tree model. 
test = OneHotEncoder(categories = X.columns)
