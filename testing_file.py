#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#dependencies
import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
directory = os.path.realpath(os.path.dirname(app_path)) 
sys.path.insert(0, directory)

import pandas as pd 
import numpy as np
import Preprocess 
from sklearn.metrics import matthews_corrcoef
from sklearn.externals import joblib 

# read test data 
data_test = pd.read_csv(os.path.join(directory, 'data/data_test.csv')) 
 
# Convert the test data to all numeric
preprocess = Preprocess.Preprocess(data_test)  
preprocess_data_test = preprocess.convertToNumeric()

# Select upper triangle of correlation matrix
corr_matrix = joblib.load(os.path.join(directory, 'corr_matrix.pkl'))  
threshold = 0.8

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 
                                  k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.7
remove_col = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop correlated features based on loaded corr_matrix
preprocess_data_test = preprocess_data_test.drop(preprocess_data_test[remove_col], axis=1)

# load the saved models 
logreg = joblib.load(os.path.join(directory, 'models/logreg.pkl'))  
forest = joblib.load(os.path.join(directory, 'models/forest.pkl'))  
gbc = joblib.load(os.path.join(directory, 'models/gbc.pkl'))  
voting_cls = joblib.load(os.path.join(directory, 'models/voting_cls.pkl'))  

# make prediction
prediction_logreg = logreg.predict(preprocess_data_test)
prediction_forest = forest.predict(preprocess_data_test)
prediction_gbc = gbc.predict(preprocess_data_test)


prediction_voting = pd.DataFrame(voting_cls.predict(preprocess_data_test))

prediction_voting.columns = ['booker']

data_test_predictions = pd.concat([preprocess_data_test['ID'], prediction_voting], axis=1)

data_test_predictions.to_csv(os.path.join(directory, 'data_test_predictions.csv'))
