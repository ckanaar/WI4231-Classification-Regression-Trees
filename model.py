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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.compose import make_column_transformer
from matplotlib import pyplot as plt 
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

# =============================================================================
# Enable plotting
plotting = False

# Prune the tree.
pruning = True 
 
# =============================================================================
# Importing and formatting the data.
data            = pd.read_csv("data/data.csv",sep = ",")
data_top        = data.columns 

y               = data.pop("Prijs")                                   # [-]. Output vector 
X               = data                                                # [-]. Design matrix

# One-hot-encoding the categorical variables 
column_trans   = make_column_transformer((OneHotEncoder(sparse = False),["Type","Shop","On_Sale","For","Known_Brand","Leather","Color"]))
X              = column_trans.fit_transform(X)

train          = 0.7                                       # [-]. Train ratio
test           = 1 - train                                 # [-]. Test ratio 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test, shuffle = False)

assert np.isclose(len(X_train)+len(X_test),len(data)), "Implementation error: Train and Test data not properly separated."

# =============================================================================
# Building the classification tree model.
def build_model_pre_pruning(X_train,y_train):
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
    model = tree.DecisionTreeRegressor()
    model = model.fit(X_train,y_train)
    
    return model 

def prune_tree(model,X_train,y_train,X_test,y_test):
    """
    Parameters
    ----------
    model : Unpruned model.
    X_train : Train design matrix
    y_train : Train output vector.
    X_test : Test design matrix
    y_test : Test output vector.

    Returns
    -------
    model : Pruned model.
    """
    path                   = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    scores                 = [] 

    for ccp_alpha in ccp_alphas: 
        model = tree.DecisionTreeRegressor(ccp_alpha=ccp_alpha)
        model = model.fit(X_train,y_train)
        score = model.score(X_test,y_test)
        scores.append(score)

    ccp_alpha_optmial = ccp_alphas[np.argmax(scores)]
    model             = tree.DecisionTreeRegressor(ccp_alpha=ccp_alpha_optmial)
    model             = model.fit(X_train,y_train)
    
    return model
    
# =============================================================================
# Executing the building and predicting. 
model_pre_pruning       = build_model_pre_pruning(X_train,y_train)
score_pre_pruning_train = model_pre_pruning.score(X_train,y_train)
score_pre_pruning_test  = model_pre_pruning.score(X_test,y_test)
if pruning == False: 
    model = model_pre_pruning

# =============================================================================
# Pruning the tree
if pruning:
    model_post_pruning        = prune_tree(model_pre_pruning,X_train,y_train,X_test,y_test)
    score_post_pruning_train  = model_post_pruning.score(X_train,y_train)
    score_post_pruning_test   = model_post_pruning.score(X_test,y_test)
    model = model_post_pruning

# =============================================================================
# Extract properties of the (un)pruned tree. 
prediction       = model.predict(X_test) 
n_leaves         = model.get_n_leaves()
n_outcomes       = len(np.unique(prediction))
model_parameters = model.get_params()

# =============================================================================
# Plotting the tree
if plotting: 
    fig_tree_prune, ax_tree_prune = plt.subplots(figsize  = (12,6), dpi = 300)
    tree.plot_tree(model)
    plt.savefig("figures/tree.png")

# =============================================================================
# Performing bagging
model_bagging = BaggingRegressor(base_estimator= None, n_estimators = 10)
model_bagging = model_bagging.fit(X_train,y_train)

score_bagging_train = model_bagging.score(X_train,y_train)
score_bagging_test  = model_bagging.score(X_test,y_test)

# =============================================================================
# Perform random forest 
model_RF = RandomForestRegressor(n_estimators = 100)
model_RF = model_RF.fit(X_train,y_train)

score_RF_train = model_RF.score(X_train,y_train)
score_RF_test  = model_RF.score(X_test,y_test)































    
    
    
 