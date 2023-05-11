# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:38:26 2019

@author: Raja Ravichandra
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
dataset = pd.read_csv("output2.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,14].values 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))