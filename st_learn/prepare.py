import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn_pandas import DataFrameMapper

import cleanup


def drop_text(X):
    return X.drop('text', axis=1)


def only_text(X):
    return X['text']


def filter_rows(df, column, values) -> pd.DataFrame:
    df_new = df.loc[df[column].isin(values), :]
    assert len(df_new[column].unique()) == len(values), "Something went wrong with filtering the rows"
    return df_new


def game_prefixer(X):
    GS = ["ACTION", "ADVENTURE", "ARCADE", "BOARD", "CARD", "CASUAL", "EDUCATIONAL",
          "PUZZLE", "RACING", "ROLE PLAYING", "SIMULATION", "SPORTS", "STRATEGY"]

    # category: identify Game categories
    #
    # for game_cat in GAME_SUBCATEGORY:
    #     data.loc[data['category'] == game_cat, ['category']] = ('GAME' + "_" + game_cat)

    # X = np.array(list(map(lambda x: 'GAME' + "_" + x if (x in GS) else x, X)))
    vfn = np.vectorize(lambda x: 'GAME-' + x if (x in GS) else x)
    return vfn(X)


def categoricals_encode(category):
    # technically the LabelEncoder is used for y only
    # should use OneHotEncoder which reqired integer values

    steps = []

    if category is None:
        # if a category is given, these features have been dropped
        steps.extend([
            ('installs', [LabelEncoder()]),
            ('contentRating', [LabelEncoder()]),
        ])

    # add category encoding
    steps.extend([
        ('category', [FunctionTransformer(game_prefixer, validate=False), LabelEncoder()])
    ])

    # pass lists of transformers for reporting to work properly
    return DataFrameMapper(steps, default=None)


def update_price_quartiles(y, q25, q50, q75):
    q100_idx = y.index
    q75_idx = y.loc[y <= q75].index
    q50_idx = y.loc[y <= q50].index
    q25_idx = y.loc[y <= q25].index

    # y.loc[q100_idx] = 'high'
    # y.loc[q75_idx] = 'mid_high'
    # y.loc[q50_idx] = 'mid_low'
    # y.loc[q25_idx] = 'low'
    y.loc[q100_idx] = 4
    y.loc[q75_idx] = 3
    y.loc[q50_idx] = 2
    y.loc[q25_idx] = 1

    return y


def price_quartiles(y_train):
    # calculate the quartiles on y_train

    q25 = np.percentile(y_train, 25)
    q50 = np.percentile(y_train, 50)
    q75 = np.percentile(y_train, 75)
    # q100 = np.percentile(y_train, 100)

    return q25, q50, q75


# unused in pipes
def prepare_data(data, category_subgroups=None):

    # category: merge subgroups
    #
    merge_categories(data, category_subgroups)

    return data


def merge_categories(df, category_subgroups):
    for subgroup in category_subgroups:
        new_category = '_'.join(subgroup)
        for category in subgroup:
            df.loc[df['category'] == category, 'category'] = new_category


# testing
def main():
    category_subgroups = [['EVENTS', 'SOCIAL', 'BEAUTY']]
    data = prepare_data(cleanup.load_data(), category_subgroups)
    print(data.dtypes)
    print(data.shape)


if __name__ == '__main__':
    main()


# for category merging
# https://github.com/amueller/introduction_to_ml_with_python/blob/master/08-conclusion.ipynb
#
#
# unused
# class CategoryMerger(BaseEstimator, TransformerMixin):
#
#     input_shape_ = None
#
#     def __init__(self, first_param=1, second_param=2):
#         self.first_param = first_param
#         self.second_param = second_param
#
#     def fit(self, X, y=None):
#         print("fitting the model right here")
#
#         X = check_array(X, ensure_2d=False)
#
#         self.input_shape_ = X.shape
#
#         # Return the transformer
#         return self
#
#     def transform(self, X):
#         # Check is fit had been called
#         check_is_fitted(self, ['input_shape_'])
#
#         # Input validation
#         X = check_array(X, ensure_2d=False)
#
#         # Check that the input is of the same shape as the one passed
#         # during fit.
#         if X.shape != self.input_shape_:
#             raise ValueError('Shape of input is different from what was seen in `fit`')
#         # do the transform
#         # testing
#         return X


# unused
# class ColumnDropper(BaseEstimator, TransformerMixin):
#
#     def __init__(self, columns=None):
#         self.columns = columns
#
#     def transform(self, X, y=None):
#         return X.drop(self.columns, axis=1)
#
#     def fit(self, X, y=None):
#         return self


# unused
# class ColumnSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, columns=None):
#         self.columns = columns
#
#     def transform(self, X, y=None):
#         return X.loc[:, self.columns]
#
#     def fit(self, X, y=None):
#         return self
