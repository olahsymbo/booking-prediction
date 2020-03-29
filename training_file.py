#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#######################################################################
# load dependencies
import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
directory = os.path.realpath(os.path.dirname(app_path)) 
sys.path.insert(0, directory)

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import Preprocess
import Classifiers
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.externals import joblib 

#######################################################################
# read booking dataset
data = pd.read_csv(os.path.join(directory, 'data/data_train.csv'))

data.info()

data['booker'].value_counts() # number of samples in each class

data.head(5)

data.describe() # show explanation of the data

data.groupby('booker')['ID'].count().plot(kind='bar')  # plot the barchart of the classes
print("corrlation of features to class 'booker' \n", data[data.columns[1:]].corr()['booker'][:])

# divide the dataset into train and validation set
X_train, X_val = train_test_split(data, test_size=0.1, random_state=42)

# Undersampling training set
train_resample = Preprocess.Preprocess(X_train)
resampled_train_data = train_resample.resampling() 


train_target = resampled_train_data['booker'] # assign the training labels

train_data = resampled_train_data.drop('booker',1) # remove the class (booker) from training data

# convert the data from categorcal to numeric
preprocess = Preprocess.Preprocess(train_data)  
train_data = preprocess.convertToNumeric()

# Undersampling validation set
val_resample = Preprocess.Preprocess(X_val)
resampled_val_data = val_resample.resampling() 

val_target = resampled_val_data['booker'] # assign the validation labels

val_data = resampled_val_data.drop('booker',1) # remove the class (booker) from validation data

# convert the data from categorcal to numeric
preprocess = Preprocess.Preprocess(val_data)  
val_data = preprocess.convertToNumeric()

# Create correlation matrix
corr_matrix = train_data.corr().abs()

joblib.dump(corr_matrix, os.path.join(directory, 'corr_matrix.pkl')) # save correlation matrix to directory

plt.figure(figsize=(8, 8))

threshold = 0.8
# visualize the correlation matrix 
sns.heatmap(corr_matrix[(corr_matrix >= threshold) | (corr_matrix < -1)], cmap='plasma', vmax=1.0, vmin=-1.0, annot=True, 
            annot_kws={"size": 6}, square=True)

# find the upper triangle of the correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 
                                           k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.7
remove_col = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

# Drop the highly  correlated features 
train_data = train_data.drop(train_data[remove_col], axis=1)

val_data = val_data.drop(val_data[remove_col], axis=1)


# Instantiate the Classifier object
classifiers = Classifiers.CrossVal(train_data, train_target)
model_dir = os.path.join(directory, 'models')

#######################################################################

# Train Logistic Regression
logreg_param = {"penalty": ['l2'],
                'C': [0.0005, 0.005, 0.05, 0.5, 0.1]}

score_logreg, logreg = classifiers.cross_val_logreg(logreg_param)

joblib.dump(logreg, os.path.join(model_dir, 'logreg.pkl')) # save logistic regression model

# Train random forest classifier
forest_params = {"criterion": ["gini", "entropy"],
                 "n_estimators": list(range(10,30,50)), 
                 "max_depth": list(range(5,10,15)), 
                 "min_samples_leaf": list(range(5,15,30))}

score_forest, forest = classifiers.cross_val_forest(forest_params)

joblib.dump(forest, os.path.join(model_dir, 'forest.pkl')) # save random forest model

# Train Gradient Boosting classifier
gbc_params = {"max_depth": list(range(5,10,15)), 
              "n_estimators": list(range(10,30,50)), 
              "min_samples_leaf": list(range(5,15,30))}
 
score_gbc, gbc = classifiers.cross_val_gbc(gbc_params) 

joblib.dump(gbc, os.path.join(model_dir, 'gbc.pkl'))   # save Gradient Boosting classifier model

#######################################################################

## Training a Voting Classifier to aggregate the three ML models
estimators=[('logreg', logreg), ('forest', forest), ('gbc', gbc)]

voting_cls = classifiers.voting_classifier(estimators)
  
joblib.dump(gbc, os.path.join(model_dir, 'voting_cls.pkl')) # save Voting Classifier

# Testing with validation set
pred_voting = voting_cls.predict(val_data)

acc = matthews_corrcoef(val_target, pred_voting)

print("Accuracy : " + str(acc)) 
 