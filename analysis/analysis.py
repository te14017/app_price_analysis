#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:50:31 2017

@author: tante, simon
"""
# templates for implementing transformers:
# https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/template.py

import re
import pandas as pd
import numpy as np
import analysis.analysisUtil as util
import cleanup.cleanup as cleanup
import cleanup.cleanupUtil as cleanUtil

# for further processing data
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_extraction.text import TfidfVectorizer

# candidate regression models
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# candidate classification models
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import MultinomialNB

# for selection and evaluation of models
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def prepare_data(priceAsRange=True, produceDocuments=False,
                  textOnly=False, dropList=None):
    # load dataframe as a whole from preprocess
    X = cleanup.load_data(produceDocuments=produceDocuments)
    if dropList is not None:
        X.drop(dropList, axis=1, inplace=True)
    
    if textOnly:
        X.dropna(subset=['text','price','price_range_index'], how='any', inplace=True)
        X.reset_index(drop=True, inplace=True)
        cols = X.columns
        pattern = re.compile(r'category_\w+')
        features = []
        for col in cols:
            match = pattern.search(col)
            if match:
                features.append(col)
        features.extend(['text','price','price_range_index'])
        X_text = X[features]
        print(X_text.dtypes)
        print(X_text.shape)
        return X_text
    else:    
        X.drop(cleanUtil.NON_NUMERIC_FEATURES, axis=1, inplace=True)
        X.drop(cleanUtil.SIMILAR_APPS_STATS, axis=1, inplace=True)
        X.dropna(subset=['text'], how='any', inplace=True)
    
        output_features = ['price']
        output_features.extend(util.FEATURES_TO_BE_SCALED)
        # produce outputs of raw data for report use
        if produceDocuments:
            util.plot_histogram(X, output_features, 'raw_')
            util.write_statistics(X, output_features, 'raw_')
        
        # outliers detection
        drop_outliers_dict = {'price':(0, 1.5)}
        #_drop_outliers(X, drop_outliers_dict)
        #if produceDocuments:
            #util.plot_histogram(X, drop_outliers_dict.keys(), 'processed_')
            #util.write_statistics(X, drop_outliers_dict.keys(), 'processed_')
        X.reset_index(drop=True, inplace=True)
        #print(X.dtypes)
        print(X.shape)
    
        return X
    
    
def post_prepare_data(X, priceAsRange, textOnly):
    if priceAsRange:
        y = X['price_range_index']
    else:
        y = X['price']
    X.drop('price_range_index', axis=1, inplace=True)
    X.drop('price', axis=1, inplace=True)
    
    # save 'name' and 'appUrl' for getting app info after Nearest Neighbor computing
    X_appInfo = None
    if not textOnly:
        X_appInfo = X[['name', 'appUrl']]
        X.drop('name', axis=1, inplace=True)
        X.drop('appUrl', axis=1, inplace=True)
    
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
    train, test = next(iter(skf.split(X, y)))
    X_train = X.loc[train,]
    X_test = X.loc[test,]
    y_train = y[train]
    y_test = y[test]
    
    if textOnly:
        X_train = X_train['text']
        X_test = X_test['text']
    else:
        X_appInfo = X_appInfo.loc[train,]
    
    return X_train, X_test, y_train, y_test, X_appInfo


def prepare_data_per_category(X):
    cols = X.columns
    # get category columns
    category_cols = []
    pattern = re.compile(r'category_\w+')
    for col in cols:
        match = pattern.search(col)
        if match:
            category_cols.append(col)
    
    categories_df = {}
    for col in category_cols:
        X_ = X.copy()
        X_.drop(X_[X_[col]!=1].index, axis=0, inplace=True)
        X_.drop(category_cols, axis=1, inplace=True)
        X_.reset_index(drop=True, inplace=True)
        categories_df[col] = X_
        
    return categories_df


def build_models(textOnly=False, priceAsRange=True):
    models = []
    # build classifiers for predicting price range
    if textOnly:
        nb = MultinomialNB()
        svm = SGDClassifier(alpha=.007, penalty='l2', loss='log')
        kNN = KNeighborsClassifier(n_neighbors=10)
        fore = RandomForestClassifier(n_estimators=500, random_state=30)
        gb = GradientBoostingClassifier(loss='deviance', learning_rate=.1, n_estimators=500,
                                        max_depth=3, criterion='friedman_mse', random_state=25)
        neural = MLPClassifier(random_state=23, hidden_layer_sizes=(10,10,10), alpha=.0001,
                               learning_rate='constant', solver='adam', momentum=.9)
        
        models.extend([nb])
        return models
    elif priceAsRange:
        svm = SGDClassifier(alpha=.01, penalty='l2')
        kNN = KNeighborsClassifier(n_neighbors=10, weights='uniform', p=2)
        nb = GaussianNB()
        tree = DecisionTreeClassifier(criterion='gini', random_state=42)
        gb = GradientBoostingClassifier(loss='deviance', learning_rate=.1, n_estimators=500,
                                        max_depth=3, criterion='friedman_mse', random_state=25)
        neural = MLPClassifier(random_state=23, hidden_layer_sizes=(10,10,10), alpha=.0001,
                               learning_rate='constant', solver='adam', momentum=.9)
    
        models.extend([gb])
    else:
    # build regressors for predicting price value
        sgd = SGDRegressor(alpha=.0001, penalty='l2')
        kNN = KNeighborsRegressor(n_neighbors=10)
        tree = DecisionTreeRegressor(random_state=53)
        neural = MLPRegressor(random_state=27, hidden_layer_sizes=(10,10,10), alpha=.0001)
        
        models.extend([sgd, kNN, tree, neural])
    
    return models


def construct_steps(textOnly=False, withTextFeature=False, priceAsRange=True):
    steps = None
    
    scaling_transformer = ScalingTransformer(featureList=util.FEATURES_TO_BE_SCALED, quantile_range=(25,75))
    selector = SelectKBest(score_func=chi2, k=15)
    if not priceAsRange:
        selector = SelectKBest(score_func=f_regression, k=15)
    to_dense = SparseToDenseTransformer()
    if textOnly:
        steps = [('tfidf', TfidfVectorizer(stop_words='english',
                                  ngram_range=(2,4))),
                #('to_dense', to_dense),
                #('selector', selector)
                ]
    elif withTextFeature:
        text_transformer = TextFeatureTransformer(predictResultAsFeature=True)
        steps = [('textTransformer', text_transformer),
                 ('scaling', scaling_transformer),
                 ('selector', selector)
                 ]
    else:
        steps = [#('outlier', dropOutliers),
                 ('scaling', scaling_transformer),
                 ('selector', selector)
                 ]
        
    return steps


class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = pd.DataFrame(data = X.toarray())
        return X_


class DropOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, featureList, produceDocuments=True, factor=1.5):
        self.produceDocuments = produceDocuments
        self.featureList = featureList
        self.factor = factor
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for f in self.featureList:
            q1 = np.percentile(X[f], 25)
            q3 = np.percentile(X[f], 75)
            # interquantile range
            iqr = q3 - q1
            floor = q1 - self.factor*iqr
            ceiling = q3 + self.factor*iqr
            outlier_bool = (X[f]<floor) | (X[f]>ceiling)
            print('amount of outliers in {f}: '.format(f=f) + str(outlier_bool.sum()))
            outlier_index = X[outlier_bool].index
            X.drop(outlier_index, axis=0, inplace=True)
        # produce outputs of processed data for report use
        if self.produceDocuments:
            util.plot_histogram(X, self.featureList, 'processed_')
            util.write_statistics(X, self.featureList, 'processed_')
        return X


def _drop_outliers(df, featureFactorDict):
    for x, factor in featureFactorDict.items():
        min_ = np.min(df[x])
        max_ = np.max(df[x])
        q1 = np.percentile(df[x], 25)
        q3 = np.percentile(df[x], 75)
        # interquantile range
        iqr = q3 - q1
        if factor[0] == 0:
            floor = min_
        else:
            floor = q1 - factor[0]*iqr
        if factor[1] == 0:
            ceiling = max_
        else:
            ceiling = q3 + factor[1]*iqr
        
        outlier_bool = (df[x]<floor) | (df[x]>ceiling)
        print('amount of outliers in {x}: '.format(x=x) + str(outlier_bool.sum()))
        outlier_index = df[outlier_bool].index
        df.drop(outlier_index, axis=0, inplace=True) 


class ScalingTransformer(BaseEstimator, TransformerMixin):
    """
    transform dataframe first by RobustScaler to lower down the influence of outliers,
    then transform it by MinMaxScaler to range (0,1)
    """
    def __init__(self, featureList, quantile_range=(25, 75)):
        # scaling continuous value features with RobustScaler
        self.quantile_range = quantile_range
        self.robust_scaler = RobustScaler(quantile_range=self.quantile_range)
        # further scaling data to range (0, 1)
        self.min_max_scaler = MinMaxScaler()
        self.featureList = featureList
            
    def fit(self, X, y=None):
        #print(X['starRating'].head())
        X_ = X.copy()
        self.robust_scaler.fit(X_[self.featureList])
        X_train_robust = self.robust_scaler.transform(X_[self.featureList])
        self.min_max_scaler.fit(X_train_robust)
        return self
        
    def transform(self, X):
        X_ = X.copy()
        X_train_robust = self.robust_scaler.transform(X_[self.featureList])
        X_[self.featureList]  = self.min_max_scaler.transform(X_train_robust)
        return X_
    

class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    transform 'text' feature in dataframe to either the text predicting result as a new feature, or
    to a dense matrix
    """
    def __init__(self, predictResultAsFeature=True):
        """
        @ predictResultAsFeature: whether use the prediction result of text feature as a new 
        feature of original data. If it's False, transfer text into tf-idf dense matrix as a new feature
        """
        self.predictResultAsFeature = predictResultAsFeature
        # countVectorizer and Tfidef transformer
        
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(2,4))
        # build classifiers for predicting result from text
        self.models = build_models(textOnly=True)
        self.best_model = self.models[0]
    
    def fit(self, X, y=None):
        X_ = X.copy()
        y_ = y
        self.tfidf.fit(X_['text'])
        text_freq = self.tfidf.transform(X_['text'])
        
        # select the best model for text feature to predict target result
        if self.predictResultAsFeature:
            self.best_model.fit(text_freq, y_)
            for model in self.models:
                model.fit(text_freq, y)
                if np.mean(model.predict(text_freq) == y_) > np.mean(self.best_model.predict(text_freq) == y_):
                    self.best_model = model
        return self
    
    def transform(self, X):
        X_ = X.copy()
        if self.predictResultAsFeature:
            text_freq = self.tfidf.transform(X_['text'])
            print("===== vocabulary size: " + str(text_freq.shape))
            # use the best model to predict probability result of transformed text data
            predict = self.best_model.predict(text_freq)
            
            X_['text'] = predict
            print("===== text predict results transformed.")
        else:
            text_freq = self.tfidf.transform(X_['text'])
            #for i, col in enumerate(self.tfidf.get_feature_names()):
                #X_[col] = pd.SparseSeries(text_freq[:, i].toarray().ravel(), fill_value=0)
            df_dense = pd.DataFrame(data = text_freq.toarray())
            X_.drop('text', axis=1, inplace=True)
            X_ = pd.concat([X_, df_dense], axis=1)
            X_.fillna(0, inplace=True)
            print("======concat dense matrix  to dataFrame done !")
        return X_
    

def build_grid(pipe, priceAsRange, textOnly):
    param_grid = []
    if priceAsRange:
        if textOnly:
            param_grid = PARAM_GRID_TEXT_MODELS
        else:
            param_grid = PARAM_GRID_CLASSIFY_MODELS
    else:
        pass
    
    grid = GridSearchCV(pipe, param_grid, scoring='accuracy', verbose=1)
    return grid
    
    
# parameters for grid search
PARAM_GRID_TEXT_NGRAM = [
        {
            'vect__ngram_range': [(3,3), (2,4), (3,5)],
            'model': [SGDClassifier(alpha=.0001, penalty='l2')],
        },
]

PARAM_GRID_TEXT_MODELS = [
        {
            'model': [SGDClassifier(alpha=.0001, penalty='l2')],
            'model__alpha' : [0.001, 0.0001, 0.01], # 0.0001
            'model__penalty' : ['l1', 'l2'], # 'l2'
        },
        {
            'model' : [KNeighborsClassifier(n_neighbors=10)],
            'model__n_neighbors' : [5, 10, 20], # 5
            'model__weights': ['uniform', 'distance'], # 'distance'
        },
]

# select between models with best parameters decided by previous grid search
PARAM_GRID_TEXT_MODEL_SELECTION = [
        {
            'model': [
                        SGDClassifier(alpha=.0001, penalty='l2'),
                        KNeighborsClassifier(n_neighbors=10, weights='distance')
                      ]
        },
]

PARAM_GRID_PREPROCESS = [
        {
            #'scaling__quantile_range' : [(25,75), (20,80)], # (25,75)
            'selector__k' : [10, 15, 30], # 30
            'selector__score_func' : [chi2, f_classif], # chi2
            'model' : [GradientBoostingClassifier(loss='deviance', learning_rate=.1, n_estimators=500,
                                        max_depth=3, criterion='friedman_mse', random_state=25)],
        },
]

PARAM_GRID_CLASSIFY_MODELS = [
        
        {
            'model': [MLPClassifier(random_state=23, alpha=.0001, learning_rate='constant',
                               learning_rate_init=.001, solver='adam', momentum=.9)],
            'model__hidden_layer_sizes': [(5,5), (100,), (100,100), (10,10,10)], # (10,10,10)
            #'model__activation': ['relu', 'logistic'],
            #'model__alpha': [0.0001, 0.001, 0.00001],
            #'model__solver': ['lbfgs', 'sgd', 'adam'],
            #'model__learning_rate': ['constant', 'adaptive', 'invscaling'],
        }
]