import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import cleanup
import evaluate
import learn
import prepare
import prediction


def priceOf(appId, df):
    p = df.loc[df['appId'] == appId, 'price']
    try:
        return float(p)
    except TypeError:
        # some appIds are not in our DB
        return np.nan


def avgPrice(sa, df):
    # sanity check
    if sa == '[]':
        return np.nan

    # remove the clutter
    sa = sa.replace('[', '')
    sa = sa.replace(']', '')
    sa = sa.replace('"', '')
    sa = sa.replace(',', ' ')

    # parse the ids
    appIds = re.findall(r'(\S*)', sa)

    # filter then map
    appPrices = [priceOf(appId, df) for appId in appIds if appId != '']

    # filter nan
    appPrices = [p for p in appPrices if ~np.isnan(p)]

    if len(appPrices) == 0:
        return np.nan
    else:
        return np.mean(appPrices)


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """
    Predicts y based on the passed idx column
    """
    def __init__(self, idx):
        self.X_ = None
        self.y_ = None
        self.idx = idx

    def fit(self, X, y):
        assert (type(self.idx) == int), "idx parameter must be integer"
        assert (self.idx < len(X.iloc[0, :])), "idx must be within range"

        # Return the classifier, no fitting
        return self

    def predict(self, X):
        # predict the value at self.idx
        # 'similarAppsAvgPrice'
        return X.iloc[:, [self.idx]].values[:, 0]


# Note that the BaselineClassifier has no training bias
# there is no training with this classifier
# also there is no searching for the Baseline
def main():
    df = cleanup.load_for_baseline()

    # full set
    X_all, y_all = cleanup.splitter(df)
    transform_eval(X_all, y_all, category=None)
    print('Done Full Set')

    # category specific
    unique_categories = df['category'].unique()
    print(unique_categories)

    for cat in unique_categories:
        print("Processing category {:}".format(cat))
        X_all, y_all = cleanup.splitter(prepare.filter_rows(df, 'category', [cat]))
        if len(X_all) < 100:
            print("Skipping category {:} with {:} samples".format(cat, len(X_all)))
        else:
            transform_eval(X_all, y_all, cat)
    print('Done Category Subgroups')


def transform_eval(X_a, y_a, category):
    X_all = X_a.copy()
    y_all = y_a.copy()

    # price quartiles - we always do this in one step with all samples
    q25, q50, q75 = prepare.price_quartiles(y_all)

    y_all = prepare.update_price_quartiles(y_all, q25, q50, q75)

    # for baseline we also need to do this for the 'similarAppsAvgPrice'
    X_all['similarAppsAvgPrice'] = prepare.update_price_quartiles(X_all['similarAppsAvgPrice'].copy(), q25, q50, q75)

    pipe = Pipeline([('baseline', BaselineClassifier(idx=list(X_all.columns).index('similarAppsAvgPrice')))])

    # learns nothing
    pipe.fit(X_all, y_all)

    y_pred = pipe.predict(X_all)

    prediction.score_predicted(y_pred, y_all, pipe, category, 'Baseline')


if __name__ == '__main__':
    main()
