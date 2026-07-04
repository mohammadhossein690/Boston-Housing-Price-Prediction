# Boston Housing Price Prediction
# Regression project using XGBoost Regressor

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
file_path = r'C:\Users\NoteBook\Desktop\Machine Learning Projects\Project 2. Boston Housing\BostonHousing.csv'
dataset = pd.read_csv(file_path)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

## Training XGBoost on the Training set
from xgboost import XGBRegressor
regressor = XGBRegressor(colsample_bytree=1.0,learning_rate= 0.03,max_depth=3,n_estimators=600,subsample=0.7)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
#----------------------------------------------------------------------------------
## Evaluating Model Performance
from sklearn import metrics
print('The r2 of y_test ',metrics.r2_score(y_test,y_pred))
print('-'*50)

## Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
r2_scores = cross_val_score(estimator=regressor,X=X_train,y=y_train,scoring='r2',cv=10)
print('The mean of r2 is',r2_scores.mean())
print('The std of r2 is',r2_scores.std())
print('-'*50)

## Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators": [200,400,600],
    "learning_rate": [0.03, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}

grid_search = GridSearchCV(estimator=regressor,param_grid=param_grid,scoring='r2',n_jobs=-1,cv=10)
grid_search.fit(X_train,y_train)
print('the best r2 is',grid_search.best_score_)
print('the best param is',grid_search.best_params_)
