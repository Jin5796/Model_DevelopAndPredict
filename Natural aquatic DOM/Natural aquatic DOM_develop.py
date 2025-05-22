# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:26:37 2022

@author: Ada
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import pearsonr

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from  sklearn.model_selection import train_test_split

ran_seed = 123
np.random.seed(ran_seed)


home = 'F:/AASecondPaper/5_Model_develop/1022New/Aquatic/'


data0 = pd.read_excel("F:/AASecondPaper/5_Model_develop/1022New/Aquatic/33aquatic.xlsx") 
data0.replace(([' NA']), np.nan, inplace=True)

#######################clean###############################
def remove_outliers(df, column):
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # 定义下界和上界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 筛选出在范围内的值
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 对于每一种化合物执行清洗操作
cleaned_df = data0.groupby('PCAS').apply(remove_outliers, column='logKdoc').reset_index(drop=True)

 
TRAIN, TEST = train_test_split(data0, test_size=0.2 ,
                              random_state=ran_seed)
 

########################impute#############################################
mean_imputer = SimpleImputer(strategy="median")

X_train = TRAIN.drop(columns=['logKdoc','PCAS'])
y_train = TRAIN['logKdoc']
X_test = TEST.drop(columns=['logKdoc','PCAS'])
y_test = TEST['logKdoc']

X_train_imputed = pd.DataFrame(mean_imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

X_test_imputed = pd.DataFrame(mean_imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

train_X_nor = X_train_imputed
train_Y_nor = y_train

test_X_nor = X_test_imputed
test_Y_nor = y_test


from sklearn.pipeline import Pipeline 

# apply Regression
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer 

from scipy.stats  import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

 

from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold

def build_models(predictors, responses, predictors_vaild, responses_vaild, modelNo):     
        
    if(modelNo==1):
        # DT
        g_cv = GridSearchCV(DecisionTreeRegressor(random_state=123),
                    param_grid = {'min_samples_split': range(2,10,2),
                                  'max_depth':range(1,10,2),
                                  'min_samples_leaf':range(1,20,2),
                                  'min_weight_fraction_leaf':[0,0.1],
                                  'max_features':range(1,50,10)
                                  }, 
                    scoring= "neg_mean_squared_error", cv=5, refit=True, n_jobs=3)
        
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "DTree";

    if(modelNo==2):
        # XGBoost      
        g_cv = GridSearchCV(XGBRegressor(random_state=123),
                param_grid = {'n_estimators':[50,100,150,200],
                                       'learning_rate':[0.1],
                                       'subsample':[0.6,0.8,1],
                                       'min_child_weight':range(1,8,2),
                                       'max_depth':range(3,15,2),
                                       },        
                scoring= "neg_mean_squared_error",n_jobs=3,cv=5, refit=True)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "XGBoost";
    
   

    if(modelNo==3):
        # GradientBoostingRegressor
        g_cv = GridSearchCV(GradientBoostingRegressor(random_state=123),
                param_grid = {'n_estimators':range(100,510,100),
                                       'learning_rate':[0.1],
                                       'subsample':[0.8,1],
                                       'max_depth':range(2,10,2),
                                       'min_samples_leaf':range(1,10,2)
                                       },        
                scoring= "neg_mean_squared_error", n_jobs=3,cv=5, refit=True)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "GBRT";

  
    if(modelNo==4):
        # RF      

       g_cv = GridSearchCV(RandomForestRegressor(random_state=123),
                param_grid = {'n_estimators':range(100,510,200),
                                       'max_depth':[None,3,5,7,9],
                                       'max_samples':[0.8,1],                                 
                                       'min_samples_split':range(2,11,4),
                                       'min_samples_leaf':range(1,5,2)
                                       },        
                scoring= "neg_mean_squared_error", n_jobs=3,cv=5, refit=True)
       g_cv.fit(predictors, responses)
       model = g_cv.best_estimator_
       modelName = "RF";

      
    model.fit(predictors, responses);
    predictions = model.predict(predictors)
    predictions_vaild = model.predict(predictors_vaild)
       
    scores = cross_val_score(model,predictors, responses,
                                 scoring="neg_mean_squared_error", cv=5,
                                 n_jobs=3)
    mse_scores = -scores
    r2_scores = cross_val_score(model, predictors, responses,
                                 scoring="r2", cv=5)
   
    Result = {};
    Result['g_cv_p'] = g_cv.best_params_
    MSE = mean_squared_error(responses,predictions)
    R2 = r2_score(responses,predictions)
    MSE_vaild = mean_squared_error(responses_vaild,predictions_vaild)
    R2_vaild = r2_score(responses_vaild,predictions_vaild)
    
    n_train = len(responses)
    k_train = predictors.shape[1]
    r2_adjusted_train = 1 - (1 - R2) * (n_train - 1) / (n_train - k_train - 1)

    n_test = len(responses_vaild)
    k_test = predictors_vaild.shape[1]
    r2_adjusted_test = 1 - (1 - R2_vaild) * (n_test - 1) / (n_test - k_test - 1)


    RMSE = np.sqrt(MSE)
    RMSE_vaild = np.sqrt(MSE_vaild)    

    Result['modelName'] = modelName;
    
    Result['model'] = model;
    
    Result['RMSEtrain'] = RMSE
    Result['R2train'] = r2_adjusted_train
    
    Result['RMSEvaild'] = RMSE_vaild
    Result['R2vaild'] = r2_adjusted_test

    Result['rmse_CV'] = np.sqrt(mse_scores.mean())
    Result['r2_CV'] = r2_scores.mean()
    
    
    return Result 

    


models = {    
    
    'DecisionTree': DecisionTreeRegressor(random_state=ran_seed),
    'XGB': XGBRegressor(random_state=ran_seed) , 
    'GBRT': GradientBoostingRegressor(random_state=ran_seed)  ,
    'RF': RandomForestRegressor(random_state=ran_seed)  
    
}

 
feature_selection_performance = {model_name: [] for model_name in models.keys()}
 
for model_name, model0 in models.items():
    print(f"Evaluating model: {model_name}")
    
    for n_features in range(1, train_X_nor.shape[1] + 1):

        rfe = RFE(estimator=model0, n_features_to_select=n_features)
        rfe.fit(train_X_nor, train_Y_nor)
        
        selected_features = train_X_nor.columns[rfe.support_]
 
        predictors = train_X_nor[selected_features]
        predictors_vaild = test_X_nor[selected_features]
        responses = train_Y_nor
        responses_vaild = test_Y_nor
        
        
        result = build_models(predictors, responses, predictors_vaild, responses_vaild,
                              modelNo=list(models.keys()).index(model_name) + 1)
        
        
        
        feature_selection_performance[model_name].append({
            'model': model_name,
            'n_features': n_features,
            'selected_features': selected_features.tolist(),
            
            # 获取模型性能
            'train_MSE' : result['MSEtrain'],
            'train_RMSE':result['RMSEtrain'],
            'train_R2' : result['R2train'],
            
            'test_MSE' : result['MSEvaild'],
            'test_RMSE ': result['RMSEvaild'],
            'test_R2' : result['R2vaild'],
            
            'CV_MSE' : result['mse_CV'],
            'CV_RMSE' : result['rmse_CV'],
            'CV_R2' : result['r2_CV'],
            'g_CV_P': result['g_cv_p']
                  
           
        })


import pandas as pd

performance_records = []
for model_name, records in feature_selection_performance.items():
    for record in records:
        record['model'] = model_name
        performance_records.append(record)

performance_df = pd.DataFrame(performance_records)   


performance_df.to_excel(home + "1115performance_Category.xlsx",index=False)





######tunning###########################
from sklearn.model_selection import LeaveOneOut

def build_models2(predictors, responses, predictors_vaild, responses_vaild, modelNo):     
    if(modelNo==3):
        # GradientBoostingRegressor
        g_cv = GridSearchCV(GradientBoostingRegressor(random_state=123),
                param_grid = {'n_estimators': range(200, 600, 50),
    'learning_rate': [0.08, 0.09, 0.1,  0.12],
    'subsample': [  0.75,   0.85,   1],
    'max_depth': range(3, 9, 1),
    'min_samples_leaf': range(5, 15, 1)
                                       },        
                scoring= "neg_mean_squared_error", n_jobs=3,cv=5, refit=True)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "GBRT";
 
      
    model.fit(predictors, responses);
    predictions = model.predict(predictors)
    predictions_vaild = model.predict(predictors_vaild)
    
    
    scores = cross_val_score(model,predictors, responses,
                                 scoring="neg_mean_squared_error", cv=5,
                                 n_jobs=3)
    mse_scores = -scores
    r2_scores = cross_val_score(model, predictors, responses,
                                 scoring="r2", cv=5)
   
    Result = {};
    Result['g_cv_p'] = g_cv.best_params_
    MSE = mean_squared_error(responses,predictions)
    R2 = r2_score(responses,predictions)
    MSE_vaild = mean_squared_error(responses_vaild,predictions_vaild)
    R2_vaild = r2_score(responses_vaild,predictions_vaild)
    
    n_train = len(responses)
    k_train = predictors.shape[1]
    r2_adjusted_train = 1 - (1 - R2) * (n_train - 1) / (n_train - k_train - 1)

    n_test = len(responses_vaild)
    k_test = predictors_vaild.shape[1]
    r2_adjusted_test = 1 - (1 - R2_vaild) * (n_test - 1) / (n_test - k_test - 1)


    RMSE = np.sqrt(MSE)
    RMSE_vaild = np.sqrt(MSE_vaild)    

    Result['modelName'] = modelName;
    
    Result['model'] = model;
    
    Result['RMSEtrain'] = RMSE
    Result['R2train'] = r2_adjusted_train
    
    Result['RMSEvaild'] = RMSE_vaild
    Result['R2vaild'] = r2_adjusted_test

    Result['rmse_CV'] = np.sqrt(mse_scores.mean())
    Result['r2_CV'] = r2_scores.mean()
    
     
    return Result 

 
selected_features2 =  ['CrippenLogP', 'SlogP_VSA1', 'MATS2m', 'PatternFP730_RDKit', 'Mor29m_R']

predictors2 = predictors[selected_features2]
predictors_vaild2 = predictors_vaild[selected_features2]

result2 = build_models2(predictors2, responses, predictors_vaild2, responses_vaild,
                      modelNo=3)

 

result_df2 = pd.DataFrame(list(result2.items()), columns=['Metric', 'Value'])
 

performance_records = []
for model_name, records in feature_selection_performance.items():
    for record in records:
        record['model'] = model_name
        performance_records.append(record)

performance_df = pd.DataFrame(performance_records)   
 



























