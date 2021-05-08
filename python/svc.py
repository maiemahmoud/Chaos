# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 05:56:25 2021

@author: ncm
"""

#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
#----------------------------------------------------

dataset = pd.read_csv('choas_without.csv')
#print(dataset.shape)
X = dataset.iloc[:, range(1, 1480)].values
y = dataset.iloc[:, 1480].values
print(X)

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle =True)
 
##----------------------------------------------------
##Applying SVC Model 
#
#'''
#sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True,
#                probability=False, tol=0.001, cache_size=200, class_weight=None,verbose=False,
#                max_iter=-1, decision_function_shape='ovr’, random_state=None)
#'''
#
#SVCModel = SVC(kernel= 'rbf',# it can be also linear,poly,sigmoid,precomputed
#               max_iter=100,C=1.0,gamma='auto')
#SVCModel.fit(X_train, y_train)
#
##Calculating Details
#print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
#print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
#print('----------------------------------------------------')
#
##Calculating Prediction
#y_pred = SVCModel.predict(X_test)
#print("-------------------")
#print (accuracy_score(y_test, y_pred)*100)
#print('Predicted Value for SVCModel is : ' , y_pred[:10])
#
##----------------------------------------------------
##Calculating Confusion Matrix
#CM = confusion_matrix(y_test, y_pred)
#print('Confusion Matrix is : \n', CM)
#
## drawing confusion matrix
#sns.heatmap(CM, center = True)
#plt.show()
#------------------------------------------------------------------------------------

###Applying MLPClassifier Model 
#from sklearn.neural_network import MLPClassifier
##----------------------------------------------------
#
#
#
#
#MLPClassifierModel = MLPClassifier(activation='tanh', # can be also identity , logistic , relu
#                                   solver='lbfgs',  # can be also sgd , adam
#                                   learning_rate='constant', # can be also invscaling , adaptive
#                                   early_stopping= False,
#                                   alpha=0.0001 ,hidden_layer_sizes=(100, 3),random_state=33)
#MLPClassifierModel.fit(X_train, y_train)
#
#
#
##Calculating Prediction
#y_pred = MLPClassifierModel.predict(X_test)
#y_pred_prob = MLPClassifierModel.predict_proba(X_test)
#print('Predicted Value for MLPClassifierModel is : ' , y_pred[:10])
#print('Prediction Probabilities Value for MLPClassifierModel is : ' , y_pred_prob[:10])
# 
#
#
##----------------------------------------------------
#
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import f1_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn.metrics import classification_report
#from sklearn.metrics import auc
#
##Calculating Confusion Matrix
#CM = confusion_matrix(y_test, y_pred)
#print('Confusion Matrix is : \n', CM)
#
## drawing confusion matrix
#sns.heatmap(CM, center = True)
#plt.show()
#
##----------------------------------------------------
##Calculating Accuracy Score  
#AccScore = accuracy_score(y_test, y_pred, normalize=False)
#print('Accuracy Score is : ', AccScore)
#
##----------------------------------------------------
##Calculating F1 Score  
#F1Score = f1_score(y_test, y_pred, average='micro') 
#print('F1 Score is : ', F1Score)
#
##----------------------------------------------------
##Calculating Recall Score 
#RecallScore = recall_score(y_test, y_pred, average='micro')
#print('Recall:',RecallScore) 
##----------------------------------------------------
##Calculating Precision Score 
#PrecisionScore = precision_score(y_test, y_pred, average='micro') 
#print('Precision Score is : ', PrecisionScore)
#
#
#
#
#
##lr_probs = MLPClassifierModel.predict_proba(X_test)
##print(lr_probs.shape)
##print(y_test.shape)
##
##from sklearn.metrics import roc_auc_score
##lr_auc = roc_auc_score(y_test, lr_probs)
### calculate AUC
##
##print('AUC: ' , lr_auc)
##
##
##
#
#
#
#
#
##
#from sklearn.metrics import mean_absolute_error
#mean_absolute_error(y_test, y_pred)
#
#from sklearn.metrics import mean_squared_error
#mean_squared_error(y_test, y_pred)
#
#from sklearn.metrics import median_absolute_error
#median_absolute_error(y_test, y_pred)
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

clf_gini.predict([[4, 4, 3, 3]])
clf_entropy.predict([[4, 4, 3, 3]])

y_pred = clf_gini.predict(X_test)
y_pred


y_pred_en = clf_entropy.predict(X_test)
y_pred_en

print ("Accuracy for gini is ", accuracy_score(y_test,y_pred)*100)

print ("Accuracy for entropy is ", accuracy_score(y_test,y_pred_en)*100)






