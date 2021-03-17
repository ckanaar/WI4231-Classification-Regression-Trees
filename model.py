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
from sklearn.compose import make_column_transformer
 
# =============================================================================
# Importing and formatting the data.
data        = pd.read_csv("data/data.csv",sep = ",")
data_top = data.columns 

y = data.pop("Prijs")                                   # [-]. Output vector 
X = data                                                # [-]. Design matrix

# One-hot-encoding the categorical variables 
column_trans = make_column_transformer((OneHotEncoder(sparse = False),["Type","Shop","On_Sale","For","Known_Brand","Leather","Color"]))
X = column_trans.fit_transform(X)

train       = 0.7                                       # [-]. Train ratio
test        = 1 - train                                 # [-]. Test ratio 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test, shuffle = False)

assert np.isclose(len(X_train)+len(X_test),len(data)), "Implementation error: Train and Test data not properly separated."

# =============================================================================
# Building the classification tree model.
def build_model(X_train,y_train):
    """
    Parameters
    ----------
    X_train : Train design matrix. 
    y_train : Train output vector. 

    Returns
    -------
    model : Regressor Tree model. 
    """   
    # Building the tree model 
    model = DecisionTreeRegressor()
    model = model.fit(X_train,y_train)
    
    return model 

# Predicting the model. 
def predict_score(X_test,y_test,model):
    """
    Parameters
    ----------
    X_test : Test Design Matrix.
    y_test : Test output vector.
    model : Regressor Tree model.

    Returns
    -------
    score : Negative MSE score of the prediction
    """
    # Predicting the test matrix. 
    prediction = model.predict(X_test)
    
    # Scoring the test matrix based on negative MSE. 
    score = sum(-(prediction - np.array(y_test))**2)/len(X_test)
    
    return score 

# =============================================================================
# Executing the building and predicting. 
model = build_model(X_train,y_train)
score = predict_score(X_test,y_test,model)


    
    
    
 