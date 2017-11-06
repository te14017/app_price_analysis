#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 00:26:42 2017

@author: tante, simon
"""

import os
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import DictWriter

from sklearn import metrics
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import learning_curve
from sklearn.externals import joblib

FEATURES_TO_BE_SCALED = ['starRating', 'days_since_lastUpdated', 'totalNrOfReviews', 'libraries', '1_star_reviews',
                         '2_star_reviews', '3_star_reviews', '4_star_reviews', '5_star_reviews',
                         'days_since_release']

OUTPUTS_PATH = os.getcwd() + "/resources/outputs/"
RAW_DATA_FOLDER = 'raw_data/'


def plot_histogram(data, featureList, prefix=''):
    # plot and save distribution histogram
    i = 0
    subFolder = ''
    if prefix == 'raw_':
        subFolder = RAW_DATA_FOLDER
    elif prefix == 'processed_':
        subFolder = 'processed_data/'
        
    for x in featureList:
        i = i+1
        fig = plt.figure()
        plt.hist(data[x], color='blue', alpha=0.5)
        plt.title("Distribution of {name}".format(name=data[x].name))
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        fig.savefig(OUTPUTS_PATH + subFolder + prefix + str(i) + '_Distribution_of_{name}.png'.format(name=data[x].name))
        plt.close(fig)
        
        
def write_statistics(data, featureList, prefix):
    headers = ['feature', 'min', 'max', 'mean', 'std', '25%', '50%', '75%']
    with open(OUTPUTS_PATH + prefix + 'numeric_stats.csv', 'w') as f:
        f_csv = DictWriter(f, headers)
        f_csv.writeheader()
        for x in featureList:
            stats = data[x].describe()
            dict_stats = {}
            dict_stats['feature'] = stats.name        
            for y in stats.index:
                if y != 'count':
                    dict_stats[y] = stats[y]
            f_csv.writerow(dict_stats)
    
    
def write_plot_valueCounts(data, featureList, prefix='category_'):
    with open(OUTPUTS_PATH + prefix + 'value_counts.txt', 'w') as f:
        for x in featureList:
            counts = data[x].value_counts()
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            counts.plot(ax=ax, kind='bar', alpha=0.5, color='grey')
            fig.savefig(OUTPUTS_PATH + RAW_DATA_FOLDER + 'value_counts_{name}.png'.format(name=data[x].name))
            plt.close(fig)
            
            f.write('\n' + x + ': \n')
            f.write(counts.to_string() + '\n')
            
            
def write_predict_results(models, X_transformed, y_train, X_test_transformed, y_test,
                          textOnly, withTextFeature, produceDocuments, priceAsRange, category=None, crossValid=False):
    for model in models:
        # plot learning curve if needed
        if crossValid:
            model_ = deepcopy(model)
            plot_learning_curve(model_, X_transformed, y_train, textOnly, withTextFeature)
        
        model.fit(X_transformed, y_train)
        
        predicted_train = model.predict(X_transformed)
        predicted = model.predict(X_test_transformed)
        
        # save models to file
        model_file = "model_{clf}_no_text.sav".format(clf=model.__class__.__name__)
        if textOnly:
            model_file = "model_{clf}_textOnly.sav".format(clf=model.__class__.__name__)
        elif withTextFeature:
            model_file = "model_{clf}_with_text.sav".format(clf=model.__class__.__name__)
        if category:
            model_file = 'per_category/' + 'model_' + category + '_' + model_file
        joblib.dump(model, OUTPUTS_PATH + model_file)
        
        if not category:
            cate = 'all categories'
        else:
            cate = category
        if priceAsRange:
            print("accuracy of {clf} (for {cate}): ".format(clf=model.__class__.__name__, cate=cate) 
            + str(np.mean(predicted_train==y_train)) + '(train), ' + str(np.mean(predicted == y_test)) + "(test)\n")
            
            #print("==== tree depth is: " + str(model.tree_.max_depth))
        else:
            print("explained_variance_score of {clf} (for {cate}): ".format(clf=model.__class__.__name__, cate=cate) 
            + str(explained_variance_score(predicted_train, y_train, multioutput='uniform_average')) + '(train), ' + 
            str(explained_variance_score(predicted, y_test, multioutput='uniform_average')) + "(test)\n")
        
    if produceDocuments and priceAsRange:
        fileName = 'clf_report_no_textFeature.txt'
        if textOnly:
            fileName = 'clf_report_textOnly.txt'
        elif withTextFeature:
            fileName = 'clf_report_withTextFeature.txt'      
        if category:
            fileName = 'per_category/' + category + '_' + fileName
        
        def _add_percent(g):
            g['percent'] = g['count']/g['count'].sum()
            return g
        
        with open(OUTPUTS_PATH + fileName, 'w') as f:
            for model in models:
                predicted = model.predict(X_test_transformed)
                f.write("\n" + model.__class__.__name__ + ": \n")
                f.write("accuracy: " + str(np.mean(predicted == y_test)) + "\n\n")
                f.write(metrics.classification_report(y_test, predicted))
                f.write("\npredict results details:\n\n")
                results = pd.DataFrame(data={'predict': predicted, 'target': y_test})
                counts_report = results.groupby(['target', 'predict']).size()
                count_df = pd.DataFrame(data={'count': counts_report})
                count_df['error_type'] = np.abs(counts_report.index.get_level_values(level=0)-counts_report.index.get_level_values(level=1))
                count_df = count_df.groupby(level=['target']).apply(_add_percent)
                count_df.sort_values('error_type', ascending=False)
                f.write(count_df.to_string())
                f.write("\n==========================================\n")
                
    elif produceDocuments and not priceAsRange:
        fileName = 'reg_report_no_textFeature.txt'
        if textOnly:
            fileName = 'reg_report_textOnly.txt'
        elif withTextFeature:
            fileName = 'reg_report_withTextFeature.txt'
        if category:
            fileName = 'per_category/' + category + '_' + fileName
        with open(OUTPUTS_PATH + fileName, 'w') as f:
            for model in models:
                predicted = model.predict(X_test_transformed)
                f.write("\n" + model.__class__.__name__ + ": \n")
                f.write("explained_variance_score of {clf} (for {cate}): ".format(clf=model.__class__.__name__, cate=cate) 
                    + str(explained_variance_score(predicted, y_test, multioutput='uniform_average')) + "\n\n")
                f.write("Sample prediction result and target: \n")
                f.write(str(np.vstack((predicted, y_test))[0:200,:]))
                
                
def write_gridSearch_result(grid, X_test, y_test, textOnly,
                            withTextFeature, priceAsRange, category=None):
    predicted = grid.predict(X_test)
    if not category:
        cate = 'all categories'
    else:
        cate = category
    print("the best model (for {cate}) is: ".format(cate=cate) + grid.best_params_['model'].__class__.__name__)
    print("the best score (for {cate}) is: ".format(cate=cate) + str(grid.best_score_))
    print("the score on test data (for {cate}) is: ".format(cate=cate) + str(np.mean(predicted == y_test)) + '\n')
    
    fileName = 'gridSearch_report_no_textFeature.txt'
    if textOnly:
        fileName = 'gridSearch_report_textOnly.txt'
    elif withTextFeature:
        fileName = 'gridSearch_report_withTextFeature.txt'
    if category:
        fileName = 'per_category/' + category + '_' + fileName
    
    with open(OUTPUTS_PATH + fileName, 'w') as f:
        f.write("\nthe best model is: " + grid.best_params_['model'].__class__.__name__ + '\n')
        f.write("the best score is: " + str(grid.best_score_) + '\n')
        f.write("the accuracy on test data is: " + str(np.mean(predicted == y_test)) + '\n\n')
        f.write(metrics.classification_report(y_test, predicted))
        f.write("\n==========================================\n")
        f.write("the best parameters are as follow: \n")
        for key, value in grid.best_params_.items():
            f.write("%s: %r" % (key, value) + '\n')


def plot_learning_curve(model, X, y, textOnly, withTextFeature):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], cv=4)
    
    model_name = model.named_steps['model'].__class__.__name__
    surfix = "without text feature"
    if textOnly:
        surfix = "text only"
    elif withTextFeature:
        surfix = "with text feature"
    title = "Learning curve of {clf} ({surfix})".format(clf=model_name, surfix=surfix)
    
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")   
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_mean = [x for x in test_scores_mean]
    test_scores_std = np.std(test_scores, axis=1)
    print("===== avg score for {clf} train: ".format(clf=model_name) + str(train_scores_mean))
    print("===== avg score for {clf} test: ".format(clf=model_name) + str(test_scores_mean))
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    fig.savefig(OUTPUTS_PATH + "/learning_curve/learning_curve_{clf}_{surfix}".format(clf=model_name, surfix=surfix))
    plt.close(fig)

