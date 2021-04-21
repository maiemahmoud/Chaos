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

#----------------------------------------------------
#choas--->1486
#chaos_withoout-->1480
#choasQRS+choas clinical _all in one array-->1620

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle =True)
 
#----------------------------------------------------
##Applying MLPClassifier Model 
from sklearn.neural_network import MLPClassifier
#----------------------------------------------------




MLPClassifierModel = MLPClassifier(activation='tanh', # can be also identity , logistic , relu
                                   solver='lbfgs',  # can be also sgd , adam
                                   learning_rate='constant', # can be also invscaling , adaptive
                                   early_stopping= False,
                                   alpha=0.0001 ,hidden_layer_sizes=(100, 3),random_state=33)
MLPClassifierModel.fit(X_train, y_train)



#Calculating Prediction
y_pred = MLPClassifierModel.predict(X_test)
y_pred_prob = MLPClassifierModel.predict_proba(X_test)
print('Predicted Value for MLPClassifierModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for MLPClassifierModel is : ' , y_pred_prob[:10])
#Calculating Details
print('MLPClassifierModel Train Score is : ' , MLPClassifierModel.score(X_train, y_train))
print('MLPClassifierModel Test Score is : ' , MLPClassifierModel.score(X_test, y_test))
print('----------------------------------------------------')

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



lr_probs = MLPClassifierModel.predict_proba(X_test)
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
