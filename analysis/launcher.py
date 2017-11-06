#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  16 18:37:31 2017

@author: tante, simon
"""

import os
import numpy as np
import pandas as pd

import analysis.analysis as anal
import analysis.analysisUtil as aUtil

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

### parameters for controling train predict process:
TEXT_ONLY = False
WITH_TEXT_FEATURE = True

PER_CATEGORY = False
PRICE_AS_RANGE = True
CROSS_VALIDATE = False
PRODUCE_DOCUMENTS = True
WITH_GRID_SEARCH = False


def main_launcher(withTextFeature=False, priceAsRange=True, produceDocuments=False,
                      textOnly=False, withGridSearch=False, perCategory=False, crossValid=False):
    # prepare dataframe X
    X = anal.prepare_data(priceAsRange=priceAsRange, produceDocuments=produceDocuments,
                            textOnly=textOnly, dropList=None)
    
    if not textOnly and not withTextFeature:
        X.drop('text', axis=1, inplace=True)
    
    if perCategory and not crossValid:
        X_categories = anal.prepare_data_per_category(X)
        for category, df in X_categories.items():
            _train_and_predict(df, withTextFeature, priceAsRange, produceDocuments,
                           textOnly, withGridSearch, category)
    else:
        category = None
        if textOnly:
            X = X[['text','price','price_range_index']]
        _train_and_predict(X, withTextFeature, priceAsRange, produceDocuments,
                           textOnly, withGridSearch, category, crossValid)
    
    
    
def _train_and_predict(X_, withTextFeature=False, priceAsRange=True, produceDocuments=False,
                      textOnly=False, withGridSearch=False, category=None, crossValid=False):
    X = X_.copy()
    
    models = anal.build_models(textOnly=textOnly, priceAsRange=priceAsRange)
    
    # post prepare data: extract y value and split data set into train and test
    X_train, X_test, y_train, y_test, X_appInfo = anal.post_prepare_data(X, priceAsRange, textOnly)
    print(X_train.dtypes)
    print(X_train.shape)
    
    if crossValid:
        pipes = _build_pipes_with_models(models, textOnly, withTextFeature)
        _cross_validate(X_train, y_train, pipes, textOnly, withTextFeature)
        return
    
    if not textOnly:
        _export_columns(X_train)
    
    if not withGridSearch:
        # if not grid search, build just one pipe for transforming data
        pipe = _build_pipe(X_train, y_train, textOnly=textOnly, priceAsRange=priceAsRange,
                         withTextFeature=withTextFeature, produceDocuments=produceDocuments)
        
        # get train and test data transformed by pipe
        pipe.fit(X_train, y_train)
        _save_pipe(pipe, category, textOnly, withTextFeature)
        
        X_transformed = pipe.transform(X_train)
        X_test_transformed = pipe.transform(X_test)
        
        # get transformed X_train and X_test for computing Nearest Neighbor
        #if X_appInfo is not None:
            #X_nn = pd.merge(X_transformed, X_appInfo, left_index=True, right_index=True)
            #X_nn.to_csv(path_or_buf=os.getcwd() + '/resources/data_for_nearest_neighbor.csv', index=False)
        
        aUtil.write_predict_results(models, X_transformed, y_train, X_test_transformed, y_test, textOnly,
                                    withTextFeature, produceDocuments, priceAsRange, category, crossValid)
    else:
        pipe = _build_pipe(X_train, y_train, model=models[0], textOnly=textOnly, priceAsRange=priceAsRange,
                         withTextFeature=withTextFeature, produceDocuments=produceDocuments)
        grid = anal.build_grid(pipe=pipe, priceAsRange=priceAsRange, textOnly=textOnly)
        grid.fit(X_train, y_train)
        aUtil.write_gridSearch_result(grid, X_test, y_test, textOnly,
                            withTextFeature, priceAsRange, category)


def _cross_validate(X, y, pipes, textOnly=False, withTextFeature=True):
    for pipe in pipes:
        X_ = X.copy()
        aUtil.plot_learning_curve(pipe, X, y, textOnly, withTextFeature)


def _build_pipe(X, y=None, model=None, textOnly=False, withTextFeature=False,
                 produceDocuments=True, priceAsRange=True):
    if textOnly:
        # steps for text feature
        steps = anal.construct_steps(textOnly=True, priceAsRange=priceAsRange)
    elif withTextFeature:
        # steps for numeric feature involving text feature transformation
        steps = anal.construct_steps(withTextFeature=True, priceAsRange=priceAsRange)
    else:
        # steps for numeric feature without text feature
        steps = anal.construct_steps(withTextFeature=False, priceAsRange=priceAsRange)
    
    # if model is passed, then add it to steps, this is for GridSearch
    if model is not None:
        steps.extend([('model', model)])
    
    pipe = Pipeline(steps)
    return pipe


def _build_pipes_with_models(models, textOnly=False, withTextFeature=False, priceAsRange=True):
    pipes = []
    if textOnly:
        # steps for text feature
        steps = anal.construct_steps(textOnly=True, priceAsRange=priceAsRange)
    elif withTextFeature:
        # steps for numeric feature involving text feature transformation
        steps = anal.construct_steps(withTextFeature=True, priceAsRange=priceAsRange)
    else:
        # steps for numeric feature without text feature
        steps = anal.construct_steps(withTextFeature=False, priceAsRange=priceAsRange)
        
    for model in models:
        steps_ = list(steps)
        steps_.extend([('model', model)])
        pipe = Pipeline(steps_)
        pipes.append(pipe)
    
    return pipes


def _save_pipe(pipe, category, textOnly, withTextFeature=True):
    pipe_file = "pipeline{cate}_withText.sav".format(cate='_'+category if category else '')
    if textOnly:
        pipe_file = 'pipeline_textOnly.sav'
    elif not withTextFeature:
        "pipeline{cate}_noText.sav".format(cate='_'+category if category else '')
    if category:
        pipe_file = 'per_category/' + pipe_file
    joblib.dump(pipe, aUtil.OUTPUTS_PATH + pipe_file)


def _export_columns(df):
    with open(os.getcwd() + '/resources/columns.txt', 'w') as f:
        f.write(str(df.columns.values.tolist()))


def main():
        main_launcher(withTextFeature=WITH_TEXT_FEATURE, priceAsRange=PRICE_AS_RANGE,
                          produceDocuments=PRODUCE_DOCUMENTS, textOnly=TEXT_ONLY,
                          withGridSearch=WITH_GRID_SEARCH, perCategory=PER_CATEGORY,
                          crossValid=CROSS_VALIDATE)


if __name__ == '__main__':
    main()