# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 22:54:47 2021

@author: ncm
"""

#Import Libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import auc
#----------------------------------------------------
#
#dataset = pd.read_csv('choasQRS+choas clinical _all in one array.csv')
##print(dataset.shape)
#X = dataset.iloc[:, range(1, 1620)].values
#y = dataset.iloc[:, 1620].values
#print(y)
#
##----------------------------------------------------
#
##----------------------------------------------------

#dataset = pd.read_csv('choas_without.csv')
##print(dataset.shape)
#X = dataset.iloc[:, range(1, 1480)].values
#y = dataset.iloc[:, 1480].values
#print(y)

#----------------------------------------------------
#----------------------------------------------------
#dataset = pd.read_csv('choas.csv')
#print(dataset.shape)
#X = dataset.iloc[:, range(1, 1486)].values
#y = dataset.iloc[:, 1486].values
#print(y)

#----------------------------------------------------
dataset = pd.read_csv('csv_result-QRStest (1).csv')
print(dataset.shape)
X = dataset.iloc[:, range(1, 75)].values
y = dataset.iloc[:, 75].values
print(y)

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.44, random_state=42, shuffle =True)
 
#----------------------------------------------------
#Applying SVC Model 

'''
sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True,
                probability=False, tol=0.001, cache_size=200, class_weight=None,verbose=False,
                max_iter=-1, decision_function_shape='ovr’, random_state=None)
'''

SVCModel = SVC(kernel= 'rbf',# it can be also linear,poly,sigmoid,precomputed
               max_iter=100,C=1.0,gamma='auto',probability=True)
SVCModel.fit(X_train, y_train)

#Calculating Details
print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = SVCModel.predict(X_test)
print("-------------------")
print (accuracy_score(y_test, y_pred)*100)
print('Predicted Value for SVCModel is : ' , y_pred[:10])

#----------------------------------------------------
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()
#------------------------------------------------------------------------------------


#Calculating F1 Score  
F1Score = f1_score(y_test, y_pred, average='micro') 
print('F1 Score is : ', F1Score)

#----------------------------------------------------
#Calculating Recall Score 
RecallScore = recall_score(y_test, y_pred, average='micro')
print('Recall:',RecallScore) 
#----------------------------------------------------
#Calculating Precision Score 
PrecisionScore = precision_score(y_test, y_pred, average='micro') 
print('Precision Score is : ', PrecisionScore)



lr_probs = SVCModel.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
print(lr_probs.shape)
print(y_test.shape)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


lr_auc = roc_auc_score(y_test, lr_probs)
# calculate AUC

print('AUC: ' , lr_auc)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))

from sklearn.metrics import median_absolute_error
print(median_absolute_error(y_test, y_pred))
