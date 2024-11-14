import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets
iris = datasets.load_iris()
x, y=iris.data[:,1:3],iris.target #taking entire data as training data

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()

######################

print("After five fold cross validation")
labels = ['Logistic Regression', 'Random Forest Model', 'Naive Bayes Model']
for clf,label in zip([clf1,clf2,clf3],labels):
    scores = model_selection.cross_val_score(clf,x,y,cv=5,scoring='accuracy')
    print('Accuracy: ',scores.mean()," for ",label)
    
'''
Accuracy:  0.9533333333333334  for  Logistic Regression
Accuracy:  0.9399999999999998  for  Random Forest Model
Accuracy:  0.9133333333333334  for  Naive Bayes Model
'''

    
voting_clf_hard=VotingClassifier(estimators=[(labels[0],clf1),
                                             (labels[1],clf2),
                                             (labels[2],clf3)],
                                 voting='hard')

voting_clf_soft=VotingClassifier(estimators=[(labels[0],clf1),
                                             (labels[1],clf2),
                                             (labels[2],clf3)],
                                 voting='soft')


labels_new=['Logistic Regression', 'Random Forest Model', 'Naive Bayes Model', 'Voting Hard', 'Voting Soft']
for clf,label in zip([clf1,clf2,clf3,voting_clf_hard,voting_clf_soft],labels_new):
    scores = model_selection.cross_val_score(clf,x,y,cv=5,scoring='accuracy')
    print('Accuracy: ',scores.mean()," for ",label)
    
'''
Accuracy:  0.9533333333333334  for  Logistic Regression
Accuracy:  0.9399999999999998  for  Random Forest Model
Accuracy:  0.9133333333333334  for  Naive Bayes Model
Accuracy:  0.9466666666666667  for  Voting Hard
Accuracy:  0.9466666666666667  for  Voting Soft
'''