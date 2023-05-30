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

# put it all back together again
concat_data = pd.concat(appended_data, axis=1)
handParseFinal = pd.concat([subset1, concat_data], axis = 1)

#write parsed version to csv
handParseFinal.to_csv('euler_parsed.csv', index=False)

