#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# dependencies
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

class Preprocess:
    
    # class constructor
    def __init__(self, data):
    
        self.data = data 
    
    # a function to convert data from categorical to numeric 
    def convertToNumeric(self):
        
        data_new = self.data  
    
        categorical_feature_mask = data_new.dtypes != int    #check which columns of data are not integer
        
        categorical_cols = data_new.columns[categorical_feature_mask].tolist() # select non integer columns
        
        data_new[categorical_cols] = data_new[categorical_cols].apply(lambda col: LabelEncoder().fit_transform(col)) # convert non integer to numeric using label encoder and add to the data
    
        return data_new
    
    def resampling(self):
        
        non_bookers = self.data[self.data['booker']==False]  # get data of class "non booker"
        
        bookers = self.data[self.data['booker']==True] # get class with "booker"
        
        booker_resampled = resample(non_bookers, replace=True, n_samples=len(bookers), random_state=27) # undersample the data to reduce the majority class
        
        resampled = pd.concat([bookers, booker_resampled]) #concatenate both bookers and undersampled non-bookers

        return resampled
        
        
    
    