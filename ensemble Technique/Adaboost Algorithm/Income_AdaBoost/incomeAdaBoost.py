# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:24:27 2024

@author: santo
"""

import pandas as pd 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
#read csv file
loan_data=pd.read_csv("E:/datascience/ensemble Technique/Adaboost Algorithm/income.csv")
loan_data.columns
loan_data.head()
#let us split the data in input and output
X=loan_data.iloc[:,0:6]
Y=loan_data.iloc[:,6]
#split the dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
#create adaboost classifier
ada_model=AdaBoostClassifier(n_estimators=100,learning_rate=1)
#n_estimators= no of weak learner
#learning_rate=,it contributes weights of weak learner by default
# train the model
model=ada_model.fit(X_train,Y_train)
#predict the result 
#train the model

Y_pred=model.predict(X_test)

print("accuracy",metrics.accuracy_score( Y_test,Y_pred))

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

ada_model=AdaBoostClassifier(learning_rate=1,n_estimators=50,base_estimator=lr)
model=ada_model.fit(X_train,Y_train)
model