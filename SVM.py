# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:22:40 2019

@author: ravip
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
dataset = pd.read_csv("feature_extraction.csv")  
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 14].values
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score 
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))    