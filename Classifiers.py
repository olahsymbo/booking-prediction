#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# dependencies
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier 

class CrossVal:
     # class constructor
    def __init__(self, train, labels):
         
        self.train = train
        self.labels = labels 
    
    # function for performing cross validation using logistic regression
    def cross_val_logreg(self, param): 
        
        search_logreg = GridSearchCV(LogisticRegression(), param) # perform grid search cross validation using logistic regressin with n parameters
        search_logreg.fit(self.train, self.labels) # fit the model
        logreg_estimator = search_logreg.best_estimator_  # search the best cross validation model
        
        score_logreg = cross_val_score(logreg_estimator, self.train, self.labels, cv=5) # get the cv folds scores
        print("Finished training Logistic Regression, cross validation score = ", score_logreg)
        
        return score_logreg, logreg_estimator
    
    # function for performing cross validation using random forest
    def cross_val_forest(self, param): 
        search_forest = GridSearchCV(RandomForestClassifier(), param)
        search_forest.fit(self.train, self.labels)
        forest_estimator = search_forest.best_estimator_
        
        score_forest = cross_val_score(forest_estimator, self.train, self.labels, cv=5)
        print("Finished training Random Forest Classifier, cross validation score = ", score_forest)
        return score_forest, forest_estimator
    
    # function for performing cross validation using gradient boosting classifier
    def cross_val_gbc(self, param): 
        search_gbc = GridSearchCV(GradientBoostingClassifier(), param)
        search_gbc.fit(self.train, self.labels)
        gbc_estimator = search_gbc.best_estimator_
        
        score_gbc  = cross_val_score(gbc_estimator, self.train, self.labels, cv=5) 
        print("Finished training Gradient Boosting Classifier, cross validation score = ", score_gbc)
        return score_gbc, gbc_estimator
    
    
    # function for aggregating classifications using voting classifier
    def voting_classifier(self, param):
        voting_cls = VotingClassifier(param, voting='hard') # declare the voting classifier object with hard voting method
        voting_cls.fit(self.train, self.labels) # fit the classifier
        return voting_cls
