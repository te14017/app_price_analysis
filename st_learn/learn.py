import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import cleanup
import evaluate
import pipes
import prepare
import prediction

# n_jobs
NJ = -1

# scoring
SCORING = ['f1_weighted', 'precision_weighted', 'recall_weighted', 'accuracy']

# n_splits
STRAT_SPLITS = 10

# n_iter
SEARCH_ITER = 1

# verbose
VERBOSE = 5

# read from console
DEBUG = False


# (pipe_steps, pipe_params)
def random_search(cv_folds, steps_params):
    return RandomizedSearchCV(estimator=Pipeline(steps_params[0]),
                              param_distributions=steps_params[1],
                              n_iter=SEARCH_ITER,
                              scoring=SCORING, refit=SCORING[0], n_jobs=NJ,
                              cv=cv_folds, verbose=VERBOSE, error_score=np.nan)


# (pipe_steps, pipe_params)
def grid_search(cv_folds, steps_params):
    if DEBUG:
        return random_search(cv_folds, steps_params)

    return GridSearchCV(estimator=Pipeline(steps_params[0]),
                        param_grid=steps_params[1],
                        scoring=SCORING, refit=SCORING[0], n_jobs=NJ,
                        cv=cv_folds, verbose=VERBOSE, error_score=np.nan)


def main():

    # Following steps from
    # https://www.datacamp.com/community/blog/scikit-learn-cheat-sheet

    train_full_data_set()
    # train_per_category()


def train_full_data_set():
    # full load
    X_all, y_all = cleanup.splitter(cleanup.load_for_ml())

    train(X_all, y_all)
    print('Done train_full_data_set')


def train_per_category():
    # load for category processing
    df = cleanup.load_for_ml_per_category()

    unique_categories = df['category'].unique()
    print(unique_categories)

    for cat in unique_categories:
        print("Processing category {:}".format(cat))
        X_all, y_all = cleanup.splitter(prepare.filter_rows(df, 'category', [cat]))
        if len(X_all) < 100:
            print("Skipping category {:} with {:} samples".format(cat, len(X_all)))
        else:
            train(X_all, y_all, cat)
    print('Done train_per_category')


def train(X_all, y_all, category=None):
    # just to be save
    X_all = X_all.copy()
    y_all = y_all.copy()

    print("{0} X_all".format(X_all.shape))

    ######################################
    # Preprocessing
    #

    # http://scikit-learn.org/stable/datasets/index.html#external-datasets
    # Categorical (or nominal) features stored as strings (common in pandas DataFrames)
    # will need converting to integers,
    # and integer categorical variables may be best exploited when encoded as one-hot variables

    # calculate price quartiles manually
    # cannot modify y with a custom transformer
    q25, q50, q75 = prepare.price_quartiles(y_all)

    y_all = prepare.update_price_quartiles(y_all, q25, q50, q75)
    print("\ny_all value counts:")
    print(y_all.value_counts() / len(y_all))

    # Maybe
    # Standardization, Normalization, Binarization,
    # Encoding Categorical Features, Imputing Missing Values

    ######################################
    # Model building and tuning
    #
    strat = StratifiedKFold(n_splits=STRAT_SPLITS)

    # split the data set according to the apps category
    # train individual classifiers on these sub groups

    p = "fd_"
    if category is not None:
        p = "cs_"

    searches = [
        # p + "numeric", "quick_numeric", grid_search(strat, pipes.quick_numeric(category))),
        # (p + "numeric", "forest_numeric", grid_search(strat, pipes.forest_numeric(category))),
        # (p + "numeric", "gradient_numeric", grid_search(strat, pipes.gradient_numeric(category))),
        (p + "all", "forest_numeric", grid_search(strat, pipes.add_text(pipes.forest_numeric(category)))),
        (p + "all", "gradient_numeric", grid_search(strat, pipes.add_text(pipes.gradient_numeric(category)))),
        # (p + "text", "bnb_text", grid_search(strat, pipes.bnb_text())),
        # (p + "text", "gradient_text", grid_search(strat, pipes.gradient_text())),
        # (p + "text", "svm_text", grid_search(strat, pipes.svm_text())),
        ("dummy", "dummy", grid_search(strat, pipes.dummy(category)))
    ]

    for model, algo, search in searches:
        ######################################
        # Model fitting
        #
        search.fit(X_all, y_all)

        ######################################
        # Model prediction
        #
        # y_pred = search.predict(X_eval)
        # y_proba = search.predict_proba(X_eval)
        # y_proba_pos = y_proba[:, 1]

        ######################################
        # Evaluate and Report Model performance
        #
        rs = evaluate.onSearch(pd.Series(), search, SCORING, STRAT_SPLITS)

        if category is not None:
            rs['category'] = category

        evaluate.printToCsv(rs, model)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Learn runner")

    arg_parser.add_argument("-d", "--debug", help="debug mode")
    args = arg_parser.parse_args()

    if args.debug is not None:
        STRAT_SPLITS = 2
        DEBUG = True

    main()
