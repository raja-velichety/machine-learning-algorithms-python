# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:06:37 2019

@author: Raja Ravichandra
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
dataset = pd.read_csv("output1.csv")
X = dataset.iloc[:, 0:14].values  
y = dataset.iloc[:, 14].values 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred)) 