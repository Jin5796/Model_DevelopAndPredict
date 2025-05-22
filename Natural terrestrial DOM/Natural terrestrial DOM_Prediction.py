# -*- coding: utf-8 -*-
"""
Created on Wed May 21 22:33:02 2025

@author: 21191
"""

#####

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 

# apply Regression
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
    
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 
 
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
 


home = 'F:/AASecondPaper/5_Model_develop/1022New/'
 
#########Natural Terrestrial DOM########
nDesc = 5 
modelname="GBRT"
model = GradientBoostingRegressor(random_state=123,n_estimators= 100,
                                  max_depth= 8, learning_rate=0.1,
                                  min_samples_leaf = 7,
                                  subsample= 0.8) #GradientBoostingRegressor

feature_selected =   ['CrippenLogP', 'ErGFP285', 'MATS2m', 'PatternFP730', 'Mor29m']

############input##################
strat_train_set = pd.read_excel(home + 'Natural terrestrial DOM0521.xlsx')
strat_test_set = pd.read_excel(home + 'Test_data_Category.xlsx')

 

####################apply model############################
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())    
])



target = 'logKdoc'
    
train_X = strat_train_set[feature_selected]
train_Y = strat_train_set[target].copy()

vaild_X = strat_test_set[feature_selected]

  
predictors0 = train_X
responses = train_Y
predictors_vaild0= vaild_X

predictors = num_pipeline.fit_transform(predictors0)
feature_names  = num_pipeline.get_feature_names_out()
predictors = pd.DataFrame(predictors, columns=feature_names)


predictors_vaild = num_pipeline.transform(predictors_vaild0)
predictors_vaild = pd.DataFrame(predictors_vaild, columns=feature_names)


model.fit(predictors, responses)
predictions = model.predict(predictors)
predictions_vaild = model.predict(predictors_vaild)###new prediction
 


