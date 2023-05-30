#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:52:00 2023

@author: d.rankin1
"""

#import numpy as np
import pandas as pd


#read in dataset
df = pd.read_csv("euler.csv")

#get num columns and rows
numCols = df.shape[1]
numRows = df.shape[0]

#subset of data that doesn't need parsed
subset1 = pd.concat([df.iloc[:, 0:3], df.iloc[:, 5:7]], axis = 1)
#subset of data that does need parsed
subset2 = pd.concat([df.iloc[:, 3:5], df.iloc[:, 7:numCols]], axis = 1)

#num columns in subset needing parsed
numColsHandParse = subset2.shape[1]
appended_data = []

#loop through each column, parsing into separate x, y, z
for x in range(numColsHandParse):
    colName = subset2.columns[x]
    handParse = subset2[colName].str.split(',',expand=True).rename(columns={0:colName+'_x', 1:colName+'_y', 2:colName+'_z'})
    appended_data.append(handParse)

concat_data = pd.concat(appended_data, axis=1)

handParseFinal = pd.concat([subset1, concat_data], axis = 1)

handParseFinal.to_csv('euler_parsed.csv', index=False)

# y_orig = data_euler.iloc[:, numCols-1]


# scaler = MinMaxScaler(feature_range=(0, 1))
# X_orig = scaler.fit_transform(X_orig)




# scores_orig = []




# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC

################CHOOSE A CLASSIFIER AND COMMENT OUT THE REST##################

#SGD Classifier - Linear Regression
#clf = SGDClassifier(loss="hinge", random_state=0)
#DecisionTreeClassifier
#clf = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=0)
#KNeighbors Classifier
#clf = KNeighborsClassifier(n_neighbors=10, weights='uniform', leaf_size=30, p=2, metric='minkowski', n_jobs=2)
#RandomForestClassifier
#clf = RandomForestClassifier(criterion="gini", max_depth=10, min_samples_split=2, n_estimators=10, random_state=1)
#SVC - SVM Classifier
#clf = SVC(C=1.0, degree=3, kernel='rbf', probability=True, random_state=None)

############################################################################


#Cross fold validation settings
#from sklearn.model_selection import ShuffleSplit
#cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


#accuracy
# scores_orig = cross_val_score(clf, X_orig, y_orig, cv=cv)
# #recall
# scores_orig_pre = cross_val_score(clf, X_orig, y_orig, cv=cv, scoring='precision_macro')
# #precision
# scores_orig_rec = cross_val_score(clf, X_orig, y_orig, cv=cv, scoring='recall_macro')
# #f1
# scores_orig_f1 = cross_val_score(clf, X_orig, y_orig, cv=cv, scoring='f1_macro')





# print("Accuracy: %0.6f (+/- %0.2f)" % (scores_orig.mean(), scores_orig.std()*2))

# print("Precision: %0.6f (+/- %0.2f)" % (scores_orig_pre.mean(), scores_orig_pre.std()*2)) 
 
# print("Recall: %0.6f (+/- %0.2f)" % (scores_orig_rec.mean(), scores_orig_rec.std()*2)) 
 
# print("F1: %0.6f (+/- %0.2f)" % (scores_orig_f1.mean(), scores_orig_f1.std()*2))

