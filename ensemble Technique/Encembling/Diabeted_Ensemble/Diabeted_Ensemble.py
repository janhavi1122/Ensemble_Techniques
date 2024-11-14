# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:18:22 2024

@author: santo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
data=pd.read_csv("E:/datascience/ensemble Technique/Adaboost Algorithm/Diabeted_Ensemble.csv")
#___________________________________________________________________________________

data.head()
data.info()
data.columns
data.isna().sum()
#___________________________________________________________________________________

target=data[' Class variable']
#___________________________________________________________________________________

sns.countplot(x=target,palette='winter')
plt.title('Diabeted')
sns.heatmap(data.corr(),cmap='YlGnBu',annot=True,fmt='.2f')
#Obervations:
 #___________________________________________________________________________________
   
sns.countplot(x=' Number of times pregnant',data=data,hue=' Class variable',palette='pastel')
plt.title("Diabeted_Ensemble")


sns.countplot(x=' Plasma glucose concentration',data=data,hue=' Class variable',palette='pastel')
plt.title("Diabeted_Ensemble")
#

sns.countplot(x=' Diastolic blood pressure',data=data,hue=' Class variable',palette='pastel')
plt.title("Diabeted_Ensemble")
#___________________________________________________________________________________

import seaborn as sns
import matplotlib.pyplot as plt
data.info()
sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(2, figsize=(20, 13))
plt.suptitle("Diabetes Ensemble")  # Corrected the function call

ax1=sns.histplot(x=' Number of times pregnant',data=data,hue=' Class variable ',kde=True,ax=ax[0],palette='winter')
ax1.set(xlabel=' Diastolic blood pressure',title=' Distribution of Twitter_hashtags')

ax2=sns.histplot(x=' Plasma glucose concentration',data=data,hue=' Class variable',kde=True,ax=ax[0],palette='viridis')
ax2.set(xlabel=' Diastolic blood pressure',title='Daibities_Ensemble')

plt.show()
#___________________________________________________________________________________

data.hist(bins=30,figsize=(20,15),color='#005b96');
#Standerdizarion: 
#Normalization: 
    
sns.boxplot(x=data[' Number of times pregnant'])   
sns.boxplot(x=data[' Plasma glucose concentration'])   
sns.boxplot(x=data['  Diastolic blood pressure'])   
sns.boxplot(x=data[' Body mass index'])   
  
#write code for Winsorizer
from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=[' Number of times pregnant', ' Plasma glucose concentration',
                         ' Diastolic blood pressure',' Body mass index'])

df_t=winsor.fit_transform(data[{' Number of times pregnant', ' Plasma glucose concentration',' Diastolic blood pressure',' Body mass index'}])
sns.boxplot(data[' Number of times pregnant'])
sns.boxplot(df_t[' Number of times pregnant'])

df_t=winsor.fit_transform(data[{' Number of times pregnant', ' Plasma glucose concentration',' Diastolic blood pressure',' Body mass index'}])
sns.boxplot(data[' Plasma glucose concentration'])
sns.boxplot(df_t[' Plasma glucose concentration'])

df_t=winsor.fit_transform(data[{' Number of times pregnant', ' Plasma glucose concentration',' Diastolic blood pressure',' Body mass index'}])
sns.boxplot(data[' Diastolic blood pressure'])
sns.boxplot(df_t[' Diastolic blood pressure'])

df_t=winsor.fit_transform(data[{' Number of times pregnant', ' Plasma glucose concentration',' Diastolic blood pressure',' Body mass index'}])
sns.boxplot(data[' Body mass index'])
sns.boxplot(df_t[' Body mass index'])

#___________________________________________________________________________________

#check skewness
skew_df =pd.DataFrame(data.select_dtypes(np.number).columns, columns=['Feature'])
skew_df['Skew'] = skew_df['Feature'].apply(lambda feature: skew(data[feature]))
skew_df['Absolute Skew'] = skew_df['Skew'].apply(abs)

skew_df['Skewed']=skew_df['Absolute Skew'].apply(lambda x: True if x>=0.5 else False)
skew_df

#fetal Charges colume is clearly skewed as we also saw in the histogram
for columns in skew_df.query("Skewed==True")['Feature'].values:
    data[columns]=np.log1p(data[columns])
    
data.head()
data1=data.copy()

data1=pd.get_dummies(data1)    
    
data1.head()

data2=data1.copy()
sc=StandardScaler()
data2[data1.select_dtypes(np.number).columns]=sc.fit_transform(data2[data1.select_dtypes(np.number).columns])
data2.drop([' Class variable'],axis=1,inplace=True)
data2.head()

#Spliiting
data_f=data2.copy()
target=data[' Class variable']
target=target.astypes(int)

X_train,X_test,y_train,y_test = train_test_split(data_f,target,stratify=target,random_state=42,test_size=0.2)

#Modeling
from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(learning_rate=0.2,n_estimators=5000)
ada_clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix
#Evaluation on Testing Data
confusion_matrix(y_test,ada_clf.predict(X_test))
accuracy_score(y_test,ada_clf.predict(X_test))

















