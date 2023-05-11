# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:57:08 2019

@author: Raja Ravichandra
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
dataset = pd.read_csv("output2.csv")
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 14].values 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)  
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(X_train, y_train.ravel())
predictions = mlp.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix as cm ,accuracy_score
print(cm(y_test,predictions))  
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))