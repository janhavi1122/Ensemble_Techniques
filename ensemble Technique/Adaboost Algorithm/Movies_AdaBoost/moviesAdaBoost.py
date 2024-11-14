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
data=pd.read_csv("E:/datascience/ensemble Technique/Adaboost Algorithm/movies.csv")
#___________________________________________________________________________________

data.head()
data.info()
data.isna().sum()
#___________________________________________________________________________________

target=data['Start_Tech_Oscar']
#___________________________________________________________________________________

sns.countplot(x=target,palette='winter')
plt.title('Oscar Rate')
sns.heatmap(data.corr(),cmap='YlGnBu',annot=True,fmt='.2f')
#Obervations:
 #___________________________________________________________________________________
   
sns.countplot(x='Genre',data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title("0 chnace based Ticket class")
#___________________________________________________________________________________

sns.countplot(x='3D_available',data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title("0 chnace based Ticket class")
#___________________________________________________________________________________

sns.countplot(x='Genre',data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title("0 chnace based Ticket class")
#___________________________________________________________________________________

sns.set_context('notebook',font_scale=1.2)
fig,ax=plt.subplots(2,figsize=(20,13))
plt.subtitle("Distribution of twitter hashtgas and collections")


ax1=sns.histplot(x='Twitter_hastags',data=data,hue='Start_Tech_Oscar',kde=True,ax=ax[0],palette='winter')
ax1.set(xlabel='Twitter_hastags',title=' Distribution of Twitter_hashtags')

ax2=sns.histplot(x='Collection',data=data,hue='Start_Tech_Oscar',kde=True,ax=ax[1],palette='viridis')
ax2.set(xlabel='Collection',title='Distribution of fire based on target varibale')

plt.show()
#___________________________________________________________________________________

data.hist(bins=30,figsize=(20,15),color='#005b96');
#Standerdizarion: 
#Normalization: 
    
sns.boxplot(x=data['Twitter_hastags'])   
sns.boxplot(x=data['Marketing expense'])   
sns.boxplot(x=data['Time_taken'])   
sns.boxplot(x=data['Avg_age_actors'])   
  
#write code for Winsorizer
from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Twitter_hastags','Marketing expense','Time_taken','Avg_age_actors'])

df_t=winsor.fit_transform(data[{'Twitter_hastags','Marketing expense','Time_taken','Avg_age_actors'}])
sns.boxplot(data['Twitter_hastags'])
sns.boxplot(df_t['Twitter_hastags'])

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
data2.drop(['Start_Tech_Oscar'],axis=1,inplace=True)
data2.head()

#Spliiting
data_f=data2.copy()
target=data['Start_Tech_Oscar']
target=target.astypes(int)

X_train,X_test,y_train,y_test = train_test_split(data_f,target,stratify=target,random_state=42,test_size=0.2)

#Modeling
from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(learning_rate=0.02,n_estimators=5000)
ada_clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix
#Evaluation on Testing Data
confusion_matrix(y_test,ada_clf.predict(X_test))
accuracy_score(y_test,ada_clf.predict(X_test))

















