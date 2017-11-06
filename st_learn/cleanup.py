import os
import re
import tarfile
from datetime import datetime
from contextlib import closing

import baseline
import inspect

import pandas as pd
import numpy as np

# Statics
APP_BRAIN_FEATURES = ['lastAppBrainCrawlTimestamp', 'ranking', 'binarySize',
                      'libraries', 'age', 'resourcePermissions', 'commentsTag']

FILE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
RESOURCES_DIR = os.path.join(FILE_DIR, '../resources/')
RESOURCES_DIR = os.path.abspath(os.path.realpath(RESOURCES_DIR))


def splitter(df) -> (pd.DataFrame, pd.DataFrame):
    # X_all, y_all
    return df.drop('price', axis=1), df['price']


def load_for_baseline() -> pd.DataFrame:
    data = load_data()
    
    # clean it up
    dropRowsWithMissingFeature(data, 'similarAppsAvgPrice', np.nan)
    
    # can be dropped now
    dropFeatures(data, ['appUrl'])

    # assert zero missing values
    for feature in list(data.columns):
        assert data[feature].isnull().sum() == 0

    return data


def load_for_ml() -> pd.DataFrame:
    data = load_data()
    
    # similarAppsAvgPrice is baseline, appUrl is reporting
    dropFeatures(data, ['similarAppsAvgPrice', 'appUrl'])

    # assert zero missing values
    for feature in list(data.columns):
        assert data[feature].isnull().sum() == 0

    return data


def load_for_ml_per_category() -> pd.DataFrame:
    data = load_for_ml()

    dropFeatures(data, ['installs', 'contentRating'])

    return data


# load dataframe from exiting csv file, preprocess when appropriate
def load_data(forceTransform=False) -> pd.DataFrame:
    pd.set_option('display.max_colwidth', 500)
    
    input_csv = os.path.join(RESOURCES_DIR, 'data.csv')
    output_csv = os.path.join(RESOURCES_DIR, 'data_transformed.csv')
    
    if forceTransform and os.path.exists(output_csv):
        os.remove(output_csv)

    if not os.path.exists(output_csv):
        ARCHIVE_NAME = 'data.tar.gz'

        print("Decompressing {0}".format(ARCHIVE_NAME))
        with closing(tarfile.open(os.path.join(RESOURCES_DIR, ARCHIVE_NAME), "r:gz")) as archive:
            archive.extractall(path=str(RESOURCES_DIR))
        print("Done Decompressing")

        select_and_clean_data(input_csv=input_csv,
                              output_csv=output_csv
                              )

    data = pd.read_csv(output_csv)

    print(data.dtypes)
    print("{0} transformed data".format(data.shape))

    return data


# Feature Extraction (including format transformations) and Missing Values
# Once this is done, it should be pretty stable
def select_and_clean_data(input_csv, output_csv,
                          bigCompanyThreshold=10):

    # read csv, < 5 bad lines
    data = pd.read_csv(input_csv, error_bad_lines=False)

    # TODO: Maybe this makes it easier regarding missing values
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer

    # Reporting before
    print(data.dtypes)
    print("{0} after read_csv".format(data.shape))


    #
    #
    # DATA SELECTION
    #

    dropRowsWithErrorsUsingAppId(data, [
        # contains problem causing name "/ ESCAPE \"
        'air.com.kongregate.mobile.games.incredibleape.escape',
        # contains problem with installs (1.0.8 or 1.0) - again some export/read_csv problem
        'com.dotemu.thelastexpress',
        'rogerfgm.frameextractorex',
        # too many missing values
        'com.maiko.xscanpet'
    ])

    dropFeatures(data, [
        '_id',
        'lastAppInfoCrawlTimestamp',
        # baseline
        # 'appId',
        'name',
        'linkName',
        'badge',
        'size',
        # reporting
        # 'appUrl',
        'userComments',
        'commentsTag',
        'lastAppBrainCrawlTimestamp',
        'currentVersion',
        'age',
        # TODO: inAppPurchase dropping: too much missing at this point
        'inAppProducts',
        'offersInAppPurchases',
        # TODO: 'permissions' & 'resourcePermissions', must be categorical or how many permissions
        'permissions',
        'resourcePermissions',
        # TODO: waiting for more app brain data
        'ranking',
        'binarySize'
    ])

    print(data.dtypes)
    print("{0} after dropping features".format(data.shape))


    #
    #
    # DATA CLEANING & FEATURE EXTRACTION
    #

    print('\n' + 'price' + (50 * '#'))
    dropRowsWithMissingFeature(data, 'price', 0.0)

    print('\n' + 'starRating' + (50 * '#'))
    dropRowsWithMissingFeature(data, 'starRating', 0.0)

    print('\n' + 'totalNrOfReviews' + (50 * '#'))
    dropRowsWithMissingFeature(data, 'totalNrOfReviews', 0.0)

    print('\n' + 'author' + (50 * '#'))
    handleAuthor(data, bigCompanyThreshold)

    print('\n' + 'lastUpdated' + (50 * '#'))
    handleLastUpdated(data)

    print('\n' + 'description, whatsNew' + (50 * '#'))
    handleTextFeatures(data, ['description', 'whatsNew'])

    print('\n' + 'installs' + (50 * '#'))
    handleInstalls(data)

    print('\n' + 'category' + (50 * '#'))
    handleCategory(data)

    print('\n' + 'contentRating' + (50 * '#'))
    handleContentRating(data)

    print('\n' + 'libraries' + (50 * '#'))
    handleLibraries(data)

    print('\n' + 'reviewsPerStarRating' + (50 * '#'))
    handleReviewsPerStarRating(data)

    print('\n' + 'requiredAndroidVersion' + (50 * '#'))
    handleRequiredAndroidVersion(data)
    
    print('\n' + 'similarApps' + (50 * '#'))
    handleSimilarApps(data)


    # TODO: Report after, always
    # aUtil.write_plot_valueCounts(data, output_features)

    # write dataframe to csv file
    if os.path.exists(output_csv):
        os.remove(output_csv)
    data.to_csv(path_or_buf=output_csv, index=False)


def handleSimilarApps(data):
    # calculate the average price of them
    data['similarAppsAvgPrice'] = data['similarApps'].apply(lambda x: baseline.avgPrice(x, data))
    data.drop(['appId', 'similarApps'], axis=1, inplace=True)


def handleRequiredAndroidVersion(data):
    # TODO: pipeline for imputing
    versionTop = data['requiredAndroidVersion'].str.extractall('^(\d.\d)').describe().loc['top', 0]

    data.loc[data['requiredAndroidVersion'] == 'Varies with device', 'requiredAndroidVersion'] = np.nan

    print("{0} missing 'requiredAndroidVersion' replaced with their top value '{1}'"
          .format(data['requiredAndroidVersion'].isnull().sum(), versionTop))

    data.loc[data['requiredAndroidVersion'].isnull(), 'requiredAndroidVersion'] = versionTop

    # Now we can parse out the numbers
    data['requiredAndroidVersion_major'] = \
        data['requiredAndroidVersion'].apply(lambda x: re.search(r'^(\d).\d', x).group(1)).astype(float)
    data['requiredAndroidVersion_minor'] = \
        data['requiredAndroidVersion'].apply(lambda x: re.search(r'^\d.(\d)', x).group(1)).astype(float)
    data.drop('requiredAndroidVersion', axis=1, inplace=True)


def handleReviewsPerStarRating(data):
    reviews = data['reviewsPerStarRating'].str.extractall(':(\d*)}')

    # missing reviewsPerStarRating, only those with 0 at every star count
    # mostly due to missing reviewsPerStarRating section on Google Play
    missing = data.loc[data['reviewsPerStarRating'] == '[{"1":0},{"2":0},{"3":0},{"4":0},{"5":0}]', :]
    print("{0} missing 'reviewsPerStarRating' replaced with their median".format(len(missing)))

    # must iterate since we CREATE features
    for index, row in data.iterrows():
        for i in range(5):
            # don't update the row directly, it might be a copy
            data.loc[index, str(i + 1) + "starReviews"] = float(reviews.loc[index, i][0])

    data.drop('reviewsPerStarRating', axis=1, inplace=True)

    # TODO: I may not need this at all - no missing values!

    # calc median based on non-missing data
    oneStarMedian = data.loc[~data.index.isin(missing.index), ['1starReviews']].median()[0]
    twoStarMedian = data.loc[~data.index.isin(missing.index), ['2starReviews']].median()[0]
    threeStarMedian = data.loc[~data.index.isin(missing.index), ['3starReviews']].median()[0]
    fourStarMedian = data.loc[~data.index.isin(missing.index), ['4starReviews']].median()[0]
    fiveStarMedian = data.loc[~data.index.isin(missing.index), ['5starReviews']].median()[0]

    # update values of missing data based on previously calcualted medians
    data.loc[data.index.isin(missing.index), ['1starReviews']] = oneStarMedian
    data.loc[data.index.isin(missing.index), ['2starReviews']] = twoStarMedian
    data.loc[data.index.isin(missing.index), ['3starReviews']] = threeStarMedian
    data.loc[data.index.isin(missing.index), ['4starReviews']] = fourStarMedian
    data.loc[data.index.isin(missing.index), ['5starReviews']] = fiveStarMedian


def handleLibraries(data):
    # TODO: pipeline for imputing
    librariesMedian = data.loc[data['libraries'].notnull(), ['libraries']].median()[0].astype(float)
    print("{0} missing 'libraries' replaced with their median '{1}'"
          .format(data['libraries'].isnull().sum(), librariesMedian))
    data.loc[data['libraries'].isnull(), 'libraries'] = str(librariesMedian)
    data['libraries'] = data['libraries'].astype(float)


def handleContentRating(data):
    # TODO: pipeline for imputing
    missingRating = 'Unrated'
    # missingRating = 'Everyone'
    # data.loc[data['contentRating'] == 'Unrated', 'contentRating'] = np.nan
    print("{0} missing 'contentRating' replaced with '{1}'"
          .format(data['contentRating'].isnull().sum(), missingRating))
    data.loc[data['contentRating'].isnull(), 'contentRating'] = missingRating


def handleCategory(data):
    # TODO: pipeline for imputing
    data['category'] = data['category'].str.upper()
    categoryTop = data['category'].describe().loc['top']
    print("{0} missing 'category' replaced with their top value '{1}'"
          .format(data['category'].isnull().sum(), categoryTop))
    data.loc[data['category'].isnull(), 'category'] = categoryTop


def handleInstalls(data):
    data.loc[data['installs'] == '500,000 - 1,000,000', 'installs'] = '500,000+'
    data.loc[data['installs'] == '1,000,000 - 5,000,000', 'installs'] = '500,000+'
    data.loc[data['installs'] == '10,000,000 - 50,000,000', 'installs'] = '500,000+'
    # data.loc[data['installs'] == '1 - 5', 'installs'] = '1 - 50'
    # data.loc[data['installs'] == '5 - 10', 'installs'] = '1 - 50'
    # data.loc[data['installs'] == '10 - 50', 'installs'] = '1 - 50'

    # TODO: pipeline for imputing
    installsTop = data['installs'].describe().loc['top']
    print("{0} missing 'installs' replaced with their top value '{1}'"
          .format(data['installs'].isnull().sum(), installsTop))
    data.loc[data['installs'].isnull(), 'installs'] = installsTop


def handleTextFeatures(data, features):
    data['text'] = data.loc[:, features].apply(lambda x: ' '.join(map(str, x)), axis=1)
    data.drop(features, axis=1, inplace=True)


def handleLastUpdated(data):
    data['daysSinceLastUpdated'] = data['lastUpdated'] \
        .apply(lambda x: (datetime.now() - datetime.strptime(str(x), '%B %d, %Y')).days).astype(float)
    data.drop('lastUpdated', axis=1, inplace=True)


def handleAuthor(data, bigCompanyThreshold):
    author_counts = data['author'].value_counts()
    bigCompanies = author_counts[author_counts > bigCompanyThreshold]
    data['bigCompany'] = data['author'].apply(lambda x: 1.0 if x in bigCompanies else 0.0)
    data.drop('author', axis=1, inplace=True)


def dropRowsWithMissingFeature(data, feature, missingValue):
    # consistency
    # data[feature] = data[feature].astype(float)

    # Report on findings
    print("Example rows with {0} equal {1}".format(feature, missingValue))
    print(data.loc[data[feature] == missingValue, 'appUrl'].head(3))
    print("Example rows with {0} equal NaN".format(feature))
    print(data.loc[data[feature] == np.nan, 'appUrl'].head(3))

    # easier handling
    data.loc[data[feature] == missingValue, feature] = np.nan

    # log
    print("{0} samples dropped due to missing ({1}) {2}".
          format(data[feature].isnull().sum(), missingValue, feature))
    # drop
    data.dropna(subset=[feature], how='any', inplace=True)


def dropFeatures(data, featureNames):
    for feature in featureNames:
        data.drop(feature, axis=1, inplace=True)

    print("{0} features dropped".format(len(featureNames)))


def dropRowsWithErrorsUsingAppId(data, appIds):
    for appId in appIds:
        data.drop(data.loc[data.appId == appId, :].index, inplace=True)

    print("{0} samples dropped due to problems with importing".format(len(appIds)))


# testing
def main():
    load_data(forceTransform=True)


if __name__ == '__main__':
    main()
