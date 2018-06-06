# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 07:27:27 2018

@author: am21381
"""

# Classification template

# Importing the libraries
import os
print(os.getcwd())
os.chdir('C:/Vadivel/Analytics/Deep learning/Deeplearning-files-training-Citi')
print(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################## DATA FETCHING AND PREPROCESSING ############################

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13] # removing row number, customer id and surname from independent variables
y = dataset.iloc[:, 13] # exited - 1 or 0 is the dependent variables

#one hot encoding on all categorical columns
X2 = pd.get_dummies(pd.DataFrame(X)) 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state = 0) #random state has the seed to random number generator

#writing training file for reference
X_train.to_csv("X_train.csv",index=False) #if this csv is already opened, close the csv file before running this line

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

################################## ANN ############################################

#importing Keras and its libraries 
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()
#Adding the input layer and firsh hidden layer to the network
classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu', input_dim = 13))
#Adding the second hidden layer to the network
classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu'))
#Adding ouput layer
classifier.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting classifier to the Training set
classifier.fit(X_train,y_train,batch_size = 10 , nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict_proba(X_test)

Test_df = pd.DataFrame({'actual':y_test,'predicted':y_pred[:,0]})
Test_df['good'] = y_test
Test_df['bad'] = 1-y_test
Test_df['Rank'] = pd.qcut(Test_df['predicted'], 10,labels=False)
Test_df['Rank'].value_counts()
grouped = Test_df.groupby('Rank',as_index = False)

#KS Computation
Final_df = pd.DataFrame()
Final_df['min_scr'] = pd.DataFrame(grouped.min().predicted)
Final_df['max_scr'] = grouped.max().predicted
Final_df['Sum_scr'] = pd.DataFrame(grouped.sum().predicted)
Final_df['bads'] = grouped.sum().bad
Final_df['goods'] = grouped.sum().good
Final_df['total'] = Final_df.bads + Final_df.goods

KS_table = (Final_df.sort_index(by = 'min_scr',ascending=False)).reset_index(drop = True)
KS_table['Cumm_goods'] = (np.round((KS_table.goods.cumsum()/KS_table.goods.sum()),4)).apply('{0:.2%}'.format)
KS_table['Cumm_bads'] = (np.round((KS_table.bads.cumsum()/KS_table.bads.sum()),4)).apply('{0:.2%}'.format)
KS_table['ks'] = np.round(((KS_table.goods / Test_df.good.sum()).cumsum() - (KS_table.bads / Test_df.bad.sum()).cumsum()), 4) * 100
flag = lambda x: '<----' if x == KS_table.ks.max() else ''
print(KS_table)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
print(roc_auc)

################################### LOGISTIC REGRESSION #############################

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
Pred_logistic = logisticRegr.predict_proba(X_test)

Test_df_logistic = pd.DataFrame({'actual':y_test,'predicted':Pred_logistic[:,1].flatten()})
Test_df_logistic['good'] = y_test
Test_df_logistic['bad'] = 1-y_test
Test_df_logistic['Rank'] = pd.qcut(Test_df_logistic['predicted'], 10,labels=False)
grouped_logistic = Test_df_logistic.groupby('Rank',as_index = False)

#KS Computation
Final_df_logistic = pd.DataFrame()
Final_df_logistic['min_scr'] = pd.DataFrame(grouped_logistic.min().predicted)
Final_df_logistic['max_scr'] = grouped_logistic.max().predicted
Final_df_logistic['Sum_scr'] = pd.DataFrame(grouped_logistic.sum().predicted)
Final_df_logistic['bads'] = grouped_logistic.sum().bad
Final_df_logistic['goods'] = grouped_logistic.sum().good
Final_df_logistic['total'] = Final_df_logistic.bads + Final_df_logistic.goods


KS_table_logistic = (Final_df_logistic.sort_index(by = 'min_scr',ascending=False)).reset_index(drop = True)
KS_table_logistic['Cumm_goods'] = (np.round((KS_table_logistic.goods.cumsum()/KS_table_logistic.goods.sum()),4)).apply('{0:.2%}'.format)
KS_table_logistic['Cumm_bads'] = (np.round((KS_table_logistic.bads.cumsum()/KS_table_logistic.bads.sum()),4)).apply('{0:.2%}'.format)
KS_table_logistic['ks'] = np.round(((KS_table_logistic.goods / Test_df_logistic.good.sum()).cumsum() - (KS_table_logistic.bads / Test_df_logistic.bad.sum()).cumsum()), 4) * 100
flag = lambda x: '<----' if x == KS_table_logistic.ks.max() else ''
print(KS_table)
print(KS_table_logistic)

roc_auc = roc_auc_score(y_test, Pred_logistic[:,1])
print(roc_auc)

########################### ANALYSIS ANN Weights and testing with sample ########

#save classifier
#classifier_copy=classifier

#getting the weights
weights = classifier.get_weights()
#weights_copy = weights

#testing one sample
sample_test=sc.transform([[667,34,5,0,2,1,0,163830.64,0,0,1,1,0]])
print(sample_test)
print(classifier.predict_proba(sample_test))


######################### Using the GridSearchCV   ##############################

import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu', input_dim = 13))
    classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu'))
    classifier.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# to make classifier a global variable 

classifier_gs = KerasClassifier(build_fn = build_classifier)
# parameters are dictionary , Gridsearch would run different combinations of these parameters usning k fold
# would return best accuracy with the best paramters
parameters = {'batch_size' : [4,8],
              'nb_epoch' : [10,20],
              'optimizer' : ['adam']}
grid_search = GridSearchCV(estimator = classifier_gs,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(X_train,y_train)

best_paramters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_
y_pred_gs = grid_search.predict_proba(X_test)
gs_scores = pd.DataFrame(grid_search.grid_scores_)

Test_df_gs = pd.DataFrame({'actual':y_test,'predicted':y_pred_gs[:,1]})
Test_df_gs['good'] = y_test
Test_df_gs['bad'] = 1-y_test
Test_df_gs['Rank'] = pd.qcut(Test_df_gs['predicted'], 10,labels=False)
grouped_gs = Test_df_gs.groupby('Rank',as_index = False)

#KS Computation
Final_df_gs = pd.DataFrame()
Final_df_gs['min_scr'] = pd.DataFrame(grouped_gs.min().predicted)
Final_df_gs['max_scr'] = grouped_gs.max().predicted
Final_df_gs['bads'] = grouped_gs.sum().bad
Final_df_gs['goods'] = grouped_gs.sum().good
Final_df_gs['total'] = Final_df_gs.bads + Final_df_gs.goods

KS_table_gs = (Final_df_gs.sort_index(by = 'min_scr',ascending=False)).reset_index(drop = True)
KS_table_gs['Cumm_goods'] = (np.round((KS_table_gs.goods.cumsum()/KS_table_gs.goods.sum()),4)).apply('{0:.2%}'.format)
KS_table_gs['Cumm_bads'] = (np.round((KS_table_gs.bads.cumsum()/KS_table_gs.bads.sum()),4)).apply('{0:.2%}'.format)
KS_table_gs['ks'] = np.round(((KS_table_gs.goods / Test_df_gs.good.sum()).cumsum() - (KS_table_gs.bads / Test_df_gs.bad.sum()).cumsum()), 4) * 100
flag = lambda x: '<----' if x == KS_table_gs.ks.max() else ''
print(KS_table_gs)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
roc_auc = roc_auc_score(y_test, classifier_gs.predict(X_test))
print(roc_auc)

#to write what is printed in console to a file
import sys
sys.stdout=open("gs_scores.txt","w") #write into a file - "a" for append
print (gs_scores)
sys.stdout.close()







