# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 07:27:27 2018

@author: am21381
"""

# Classification template

# Importing the libraries
import os
print(os.getcwd())
os.chdir('C:\\Users\\am21381\\Desktop\\Demo\\Artificial_Neural_Networks_demo')
print(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

print (X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
#index of the independent variable (starting 0)
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
print (X)
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
print (X)
#Required only for country Index = 1
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
print (X)

X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





#importing Keras and its libraries 
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()
#Adding the input layer and firsh hidden layer to the network
classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu', input_dim = 11))
#Adding the second hidden layer to the network
classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu'))
#Adding ouput layer
classifier.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting classifier to the Training set
classifier.fit(X_train,y_train,batch_size = 10 , nb_epoch = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


Test_df = pd.DataFrame({'actual':y_test,'predicted':y_pred.flatten()})
Test_df['good'] = y_test
Test_df['bad'] = 1-y_test
Test_df['Rank'] = pd.qcut(Test_df['predicted'], 10,labels=False)
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



# k- fold cross validation
import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu'))
    classifier.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# to make classifier a global variable 

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10 , nb_epoch = 100)
accuracies = cross_val_score (estimator = classifier, X = X_train , y = y_train , cv =10, n_jobs = -1 )
mean = accuracies.mean()
variance = accuracies.std()

# Using the GridSearchCV

import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6,init = 'uniform' , activation = 'relu'))
    classifier.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# to make classifier a global variable 

classifier = KerasClassifier(build_fn = build_classifier)
# parameters are dictionary , Gridsearch would run different combinations of these parameters usning k fold
# would return best accuracy with the best paramters
parameters = {'batch_size' : [25,32],
              'nb_epoch' : [100,500],
              'optimizer' : ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train,y_train)

best_paramters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_
y_pred = grid_search.predict_proba(X_test)


Test_df = pd.DataFrame({'actual':y_test,'predicted':y_pred.flatten()})
Test_df['good'] = y_test
Test_df['bad'] = 1-y_test
Test_df['Rank'] = pd.qcut(Test_df['predicted'], 10,labels=False)
grouped = Test_df.groupby('Rank',as_index = False)

#KS Computation
Final_df = pd.DataFrame()
Final_df['min_scr'] = pd.DataFrame(grouped.min().predicted)
Final_df['max_scr'] = grouped.max().predicted
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




#store the best results













