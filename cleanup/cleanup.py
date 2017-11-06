#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:50:31 2017

@author: tante, simon
"""

import os
import csv
import ast
import re
from datetime import datetime

import pandas as pd
import numpy as np

import cleanup.cleanupUtil as util
import analysis.analysisUtil as aUtil

MERGE_LIST = None

# load dataframe from exiting csv file or load and clean up new dataframe
def load_data(from_csv = True, gameAsOneCategory=False,
              produceDocuments=False):
    if from_csv:
        return pd.read_csv(os.getcwd()+"/resources/data_transformed.csv")
    else:
        return load_clean_data(gameAsOneCategory=gameAsOneCategory,
                               produceDocuments=produceDocuments)


def load_clean_data(csv_file=os.getcwd() + "/resources/data_9_20.csv",
                  output_file=os.getcwd()+"/resources/data_transformed.csv",
                  gameAsOneCategory=False, produceDocuments=True,
                  dropNullAppbrain=True, categorizeNumeric=True, bigCompanyThreshold=1,
                  mergeList=None):
    """Return the pandas dataframe of loaded csv file, with tranformed numeric features,
    dummly values of categorized features, and text features

    The result dataframe consists of numeric features, non-numeric features specified in 
    NON_NUMERIC_FEATURES list for identification of apps, and text features in TEXT_FEATURES
    list which will be further processed in data analysis by sklearn.

    @gameAsOneCategory: whether group game subcategories together as one 'game' category
    @produceDocuments: produce output documents for report use
    @dropNullAppbrain: whether drop rows with incomplete info from AppBrain
    @categorizeNumeric: whether turn some numeric features into range and treat as categorized feature
    @
    """
    
    data = pd.read_csv(csv_file, error_bad_lines=False)
               
    # drop features which are not needed
    data.drop('_id', axis=1, inplace=True)
    data.drop('lastAppInfoCrawlTimestamp', axis=1, inplace=True)
    data.drop('lastAppBrainCrawlTimestamp', axis=1, inplace=True)
    data.drop('linkName', axis=1, inplace=True)
    data.drop('badge', axis=1, inplace=True)
    data.drop('size', axis=1, inplace=True)
    # Do not drop appUrl and Name in order to get app info after Nearest Neighbor computing
    #data.drop('appUrl', axis=1, inplace=True)
    
    # drop rows with incomplete info from AppBrain
    #if dropNullAppbrain:
        #data.dropna(subset=['binarySize'], how='any', inplace=True)
    
    #data = data.loc[data['category']=='Sports', :]
    
    # price
    # drop rows which have no price information
    data.dropna(subset=['price'], how='any', inplace=True)
    # add a price_range_index feature as the category index
    _transform_price(data)
    aUtil.plot_histogram(data, ['price', 'price_range_index'], '')
    
    # starRating
    # transform starRating to float and value 0.0 to median value
    data['starRating'] = data['starRating'].apply(_transform_to_float)
    data.loc[data['starRating'] == 0.0, ['starRating']] = data['starRating'].median()
    
    # author
    # the app is provided by a big company if this company produces apps more than the threshold
    bigCompanies = _get_bigCompanies(data, bigCompanyThreshold)
    data['bigProvider'] = data['author'].apply(lambda x: 1 if x in bigCompanies else 0)
    
    # totalNrOfReviews
    # transform into float
    # store the range temporarily since 'totalNrOfReviews' will be used next
    reviews_range = data['totalNrOfReviews'].apply(_transform_totalReviews)
    data['totalNrOfReviews'] = data['totalNrOfReviews'].apply(_transform_to_float)
    
    # authors
    _drop_abnormal_by_length(data, 'author', 80)
    
    # category
    # combine categories with too few apps into other categories according to value counts
    _transform_category(data)
    
    # category
    # if true, treat all game subcategories as one category
    if (gameAsOneCategory == True):
        data.loc[data['category'].isin(util.GAME_SUBCATEGORY), ['category']] = 'game'
        output_file=os.getcwd()+"/resources/data_game_as_one.csv"
        
    # reviewsPerStarRating            
    # divide reviewsPerStarRating into 5 separate features
    _drop_abnormal_by_length(data, 'reviewsPerStarRating', 80)
    reviews = data['reviewsPerStarRating'].apply(_transform_reviewsPerStar)           
    for i in range(5):
        per_star_reviews = 0 if reviews is None else reviews.apply(lambda x:x[i]).apply(lambda x:x.get(str(i+1)))
        data[str(i+1)+"_star_reviews"] = np.divide(per_star_reviews, data['totalNrOfReviews'])
        data.loc[data['totalNrOfReviews'] == 0.0, str(i+1)+"_star_reviews"] = 0.0
    data.drop('reviewsPerStarRating', axis=1, inplace=True)
    
    if categorizeNumeric:
        data['totalNrOfReviews'] = reviews_range
    
    # lastUpdated           
    # transform lastUpdated time into days since lastUpdated time
    data['days_since_lastUpdated'] = data['lastUpdated'].apply(_transform_lastUpdated)
    data.loc[data['days_since_lastUpdated']==0, 'days_since_lastUpdated'] = data['days_since_lastUpdated'].mean()
    data.drop('lastUpdated', axis=1, inplace=True)
    
    # installs
    # divide installs into min installs and max installs
    data.loc[data['installs'].isnull(), 'installs'] = '1 - 5'
    #data['installs_min'] = min_max.apply(lambda x:x[0])
    #data['installs_max'] = min_max.apply(lambda x:x[1])
    #data.drop('installs', axis=1, inplace=True)
     
    # currentVersion           
    data.drop('currentVersion', axis=1, inplace=True)
     
    # requiredAndroidVersion
    # divide required android version into 3 individual digits
    version_nrs = data['requiredAndroidVersion'].apply(_transform_version)
    data['requiredAndroidVersion_d1'] = version_nrs.apply(lambda x:x[0])
    data['requiredAndroidVersion_d2'] = version_nrs.apply(lambda x:x[1])
    data['requiredAndroidVersion_d3'] = version_nrs.apply(lambda x:x[2])
    # fill nan with most frequent value
    data.loc[data['requiredAndroidVersion_d1'].isnull(), ['requiredAndroidVersion_d1']] = data['requiredAndroidVersion_d1'].value_counts().index[0]
    data.loc[data['requiredAndroidVersion_d2'].isnull(), ['requiredAndroidVersion_d2']] = data['requiredAndroidVersion_d2'].value_counts().index[0]
    data.loc[data['requiredAndroidVersion_d3'].isnull(), ['requiredAndroidVersion_d3']] = data['requiredAndroidVersion_d3'].value_counts().index[0]
    data.drop('requiredAndroidVersion', axis=1, inplace=True)
    
    # contentRating
    _drop_abnormal_by_length(data, 'contentRating', 30)
     
    # inAppProducts           
    # divide inAppPurchase into min and max values
    inApp_min_max = data['inAppProducts'].apply(_transform_inAppPurchase)
    data['inAppPurchase_min'] = inApp_min_max.apply(lambda x:x[0])
    data['inAppPurchase_max'] = inApp_min_max.apply(lambda x:x[1])
    data.drop('inAppProducts', axis=1, inplace=True)
    # drop this feature currently
    data.drop('inAppPurchase_min', axis=1, inplace=True)
    data.drop('inAppPurchase_max', axis=1, inplace=True)
    
    # offersInAppPurchases
    data['offersInAppPurchases'] = data['offersInAppPurchases'].apply(lambda x: 1 if x=='Offers in-app purchases' else 0)

    # similarApps
    # transform similarApps into new features of count, mean, std, min, 25%, 50%, 75%, max of their prices
    apps_stats = data['similarApps'].apply(_transform_similarApps, df=data)
    similarApps_features = util.SIMILAR_APPS_STATS
    for i in range(8):
        data[similarApps_features[i]] = apps_stats.apply(lambda x:x[i])
        # fill nan with mean value
        data.loc[data[similarApps_features[i]].isnull(), similarApps_features[i]] = data[similarApps_features[i]].mean()
     
    # ranking           
    data.loc[:, 'ranking'] = data['ranking'].apply(_transform_ranking)
     
    # binarySize
    # transform binarySize to be measured by KB and fill 0.0 with mean value
    data.loc[:, 'binarySize'] = data['binarySize'].apply(_transform_binarySize)
    if categorizeNumeric:
        data.loc[:, 'binarySize'] = data['binarySize'].apply(_transform_binarySize_to_range)
     
    # libraries
    data['libraries'] = data['libraries'].apply(_transform_to_float)
    data['libraries'].fillna(0.0, inplace=True)
    if categorizeNumeric:
        data.loc[data['libraries']>0.0, 'libraries'] = 1.0
    
    # age
    # transform age into days since release
    data['days_since_release'] = data['age'].apply(_transform_age)
    data.loc[data['days_since_release'].isnull(), 'days_since_release'] = data['days_since_release'].mean()
    data.drop('age', axis=1, inplace=True)
    
    # text features
    # concat text features together
    _transform_text(data, util.TEXT_FEATURES)
    
    dummy_list = util.DUMMY_LIST
    if categorizeNumeric:
        dummy_list.extend(util.NUMERIC_TO_BE_CATEGORIZED)
    
    if produceDocuments:
        output_features = ['price_range_index']
        output_features.extend(dummy_list)
        aUtil.write_plot_valueCounts(data, output_features)
     
    # get dummy values for categorized features
    data = _to_dummy(data, dummy_list)
    
    # write dataframe to csv file
    try:
        os.remove(output_file)
    except OSError as e:
        print ("Failed with: " + e.strerror)

    data.to_csv(path_or_buf=output_file, index=False)
     
    return data
     

def _drop_abnormal_by_length(data, col, length):
    # drop rows which has abnormal length of values
    data.drop(data[data[col].str.len() > length].index, axis=0, inplace=True)


def _transform_price(df):
    # transform price into range index
    q25 = np.percentile(df['price'], 25)
    q50 = np.percentile(df['price'], 50)
    q75 = np.percentile(df['price'], 75)
    print("=== quantile values of price: " + ", ".join([str(q25), str(q50), str(q75)]))
    
    q100_idx = df.index
    q25_idx = df.loc[df['price'] <= q25, :].index
    q50_idx = df.loc[df['price'] <= q50, :].index
    q75_idx = df.loc[df['price'] <= q75, :].index
    
    df['price_range_index'] = 0
    df.loc[q100_idx, 'price_range_index'] = 4
    df.loc[q75_idx, 'price_range_index'] = 3
    df.loc[q50_idx, 'price_range_index'] = 2
    df.loc[q25_idx, 'price_range_index'] = 1
    print(df['price_range_index'].value_counts())
    

def _transform_to_float(string):
    try:
        number = float(string)
    except ValueError:
        number = 0.0
    return number
    

def _get_bigCompanies(df, threshold):
    counts = df['author'].value_counts()
    big_company_list = counts[(counts > threshold)==True].index
    transformed_list = [x.encode('utf-8').strip() for x in big_company_list.tolist()]
    
    with open(os.getcwd() + '/resources/bigCompanies.txt', 'w') as f:
        f.write(str(transformed_list))
    
    return big_company_list


def _transform_totalReviews(totalReviews):
    reviews = str(totalReviews)
    if (reviews == 'nan' or not reviews.strip() or reviews.strip() == "N/A"):
        reviews_range = 1
        return reviews_range
    try:
        reviews = float(reviews)
        if reviews < 100:
            reviews_range = 1
        elif reviews >=100 and reviews < 500:
            reviews_range = 2
        else:
            reviews_range = 3
    except ValueError:
        reviews_range = 1
    
    return reviews_range


def _transform_category(df, mergeList=None):
    df.loc[df['category'] == 'Beauty', 'category'] = 'Lifestyle'
    df.loc[df['category'] == 'Events', 'category'] = 'Social'
    df.loc[df['category'] == 'Dating', 'category'] = 'Social'
    df.loc[df['category'] == 'House & Home', 'category'] = 'Lifestyle'
    df.loc[df['category'] == 'Parenting', 'category'] = 'Medical'
    df.loc[df['category'] == 'Comics', 'category'] = 'Entertainment'
    df.loc[df['category'] == 'Libraries & Demo', 'category'] = 'Tools'
    df.loc[df['category'] == 'Trivia', 'category'] = 'Puzzle'
    df.loc[df['category'] == 'Art & Design', 'category'] = 'Entertainment'
    df.loc[df['category'] == 'Shopping', 'category'] = 'Lifestyle'
    df.loc[df['category'] == 'Food & Drink', 'category'] = 'Lifestyle'
    df.loc[df['category'] == 'News & Magazines', 'category'] = 'Books & Reference'
    df.loc[df['category'] == 'Auto & Vehicles', 'category'] = 'Lifestyle'
    df.loc[df['category'] == 'Casino', 'category'] = 'Puzzle'
    df.loc[df['category'] == 'Music', 'category'] = 'Educational'
    df.loc[df['category'] == 'Word', 'category'] = 'Puzzle'
    
    if mergeList is not None:
        for mergeTuple in mergeList:
            new_category = '_'.join(mergeTuple)
            for category in mergeTuple:
                df.loc[df['category'] == category, 'category'] = new_category
        

def _transform_reviewsPerStar(reviewsString):
    if (reviewsString is None or not reviewsString.strip()):
        return None
    else:
        # reviews has structure like below:
        # "[{""1"":578440},{""2"":411470},{""3"":32410},{""4"":2962},{""5"":11951}]"
        return ast.literal_eval(reviewsString)


# transfer date into days from the date to today
# lastUpdated time is like "September 4, 2016"
def _transform_lastUpdated(date):
    date = str(date)
    if (date is None or not date or date == 'nan'):
        return 0
    else:
        return (datetime.now()-datetime.strptime(date, '%B %d, %Y')).days


# installs is like: "500,000 - 1,000,000"
def _transform_installs(installs):
    installs = str(installs)
    if (installs is None or not installs.strip() or installs == 'nan'):
        return 0, 0
    else:
        inst = installs.split("-")
        if len(inst) == 1:
            try:
                min = int(inst[0].strip().replace(',',''))
                max = min
            except:
                min, max = 0, 0
        elif len(inst) == 2:
            try:
                min, max = int(inst[0].strip().replace(',','')), int(inst[1].strip().replace(',',''))
            except:
                min, max = 0, 0
        else:
            min, max = 0, 0
        return min, max


def _transform_version(version):
    version = str(version)
    if (version is None or not version.strip()):
        return None, None, None
    else:
    # TODO: whether transform version into numeric numbers
        numbers = re.findall('\d+', version)
        if (len(numbers) == 0):
            return None, None, None
        elif (len(numbers) == 1):
            return numbers[0], 0, 0
        elif (len(numbers) == 2):
            return numbers[0], numbers[1], 0
        else:
            return numbers[0], numbers[1], numbers[2]


# inAppProducts is in the form of "$0.99 - $49.99 per item"
def _transform_inAppPurchase(inApp):
    inApp = str(inApp)
    if (inApp is None or not inApp.strip()):
        return 0.0, 0.0
    else:
        inAppPrice = re.findall('\d+.\d+', inApp)
        if (len(inAppPrice) == 0):
            return 0.0, 0.0
        elif (len(inAppPrice) == 1):
            return float(inAppPrice[0]), float(inAppPrice[0])
        else:
            return float(inAppPrice[0]), float(inAppPrice[1])
        

def _transform_similarApps(apps, df):
    # return count, mean, std, min, 25%, 50%, 75%, max of similar apps prices
    apps = str(apps)
    if (apps is None or not apps.strip()):
        return None, None, None, None, None, None, None, None
    else:
        apps = ast.literal_eval(apps)
        if not isinstance(apps, list):
            return None, None, None, None, None, None, None, None
        else:
            apps_prices = df.loc[df['appId'].isin(apps), 'price']
            return apps_prices.size, apps_prices.mean(), apps_prices.std(), apps_prices.min(),\
                    apps_prices.quantile(q=0.25), apps_prices.median(),apps_prices.quantile(q=0.75), apps_prices.max()

 
def _transform_ranking(ranking):
    ranking = str(ranking)
    if (ranking == 'nan'):
        return 'no_info'
    elif (not ranking.strip()):
        return 'other'
    else:
        # TODO: transform into integer values
        return '_'.join(ranking.strip().split())


# transform all binarySize to be measured by KB
def _transform_binarySize(size):
    size = str(size)
    if (size == 'nan' or not size.strip() or size.strip() == "N/A"):
        return 0.0
    else:
        sizeNumber = re.findall('\d+.\d+', size)
        sizeUnit = re.findall('[a-zA-Z]+', size)
        if (sizeNumber == None or sizeUnit == None or len(sizeNumber) == 0 or len(sizeUnit) == 0):
            return 0.0
        elif (sizeUnit[0].upper() == 'KB'):
            return float(sizeNumber[0])
        elif (sizeUnit[0].upper() == 'MB'):
            return float(sizeNumber[0])*1024
        elif (sizeUnit[0].upper() == 'GB'):
            return float(sizeNumber[0])*1024*1024
        else:
            return 0.0


def _transform_binarySize_to_range(size):
    size = str(size)
    if (size == 'nan' or not size.strip() or size.strip() == "N/A"):
        size_range = 1
        return size_range
    try:
        size = float(size)
        if size < 3000.0:
            size_range = 1
        elif size >= 3000.0 and size <= 10000.0:
            size_range = 2
        elif size > 10000.0 and size <= 35000.0:
            size_range = 3
        else:
            size_range = 4
    except ValueError:
        size_range = 1
    
    return size_range

        
# age has the form like 2014 November
def _transform_age(age):
    age = str(age)
    if (age == 'nan' or not age.strip()):
        return None
    else:
        try:
            ageTime = datetime.strptime(age, '%Y %B')
        except ValueError:
            return None
        return (datetime.now()-ageTime).days
    
    
def _transform_text(df, texts):
    i = 1
    while (i < len(texts)):
        df[texts[i]] = df[texts[i]].map(str) + df[texts[i-1]]
        i = i+1
    df['text'] = df[texts[i-1]]
    for x in texts:
        if x == 'name':
            continue
    drop_list = util.TEXT_FEATURES
    drop_list.remove('name')
    df.drop(drop_list, axis=1, inplace=True)
    
    
def _to_dummy(df, dummy_list):
    for x in dummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df.drop(x, axis=1, inplace=True)
        df = pd.concat([df, dummies], axis=1)
    return df


## for test
def main():
    data = load_clean_data(dropNullAppbrain=True, categorizeNumeric=False, mergeList=MERGE_LIST)
    print(data.dtypes)
    print(data.shape)


if __name__ == '__main__':
    main()