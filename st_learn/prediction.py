import inspect
import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

import cleanup
import prepare
import pipes

# Statics
FILE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
RESOURCES_DIR = os.path.join(FILE_DIR, '../resources/')
SCOR_DIR = os.path.join(RESOURCES_DIR, './scoring/')


def cross_val_predict_score(X_all, y_all, model, category, name, cv):
    y_pred = cross_val_predict(model, X_all, y_all, cv=cv, n_jobs=-1)
    score_predicted(y_pred, y_all, model, category, name)


def score_predicted(y_p, y_e, model, category, name):
    y_pred = y_p.copy()
    y_eval = y_e.copy()

    diff = y_pred - y_eval
    error = round(np.sqrt((np.sum(np.power(diff, 2)) / y_pred.size)), 3)

    if category is not None:
        cat = category
    else:
        cat = 'all'

    csv = os.path.join(SCOR_DIR, 'scoring.csv')
    df = pd.read_csv(csv, error_bad_lines=False, index_col=0)

    f1 = f1_score(y_eval, y_pred, average='weighted')

    df.loc[df['Category'] == cat, name + "___f1"] = f1
    df.loc[df['Category'] == cat, name + "___prec"] = precision_score(y_eval, y_pred, average='weighted')
    df.loc[df['Category'] == cat, name + "___rec"] = recall_score(y_eval, y_pred, average='weighted')
    df.loc[df['Category'] == cat, name + "___acc"] = accuracy_score(y_eval, y_pred)
    df.loc[df['Category'] == cat, name + "___err"] = error
    df.loc[df['Category'] == cat, name + "___combined"] = "{} ({})".format(round(f1, 3), np.around(error, 2))

    df.to_csv(path_or_buf=csv, index=True)
    ydf = pd.DataFrame()
    ydf['y_eval'] = y_eval
    ydf['y_pred'] = y_pred
    ydf['diff'] = diff

    model_dir = os.path.join(SCOR_DIR, './' + name + '/')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    ydf.to_csv(path_or_buf=os.path.join(model_dir, name + '__' + cat + '_y.csv'), index=True)
    joblib.dump(model, os.path.join(model_dir, name + '__' + cat + '.pkl'))


def main():
    df = cleanup.load_for_ml()

    # full set
    X_all, y_all = cleanup.splitter(df)
    transform_eval(X_all, y_all, category=None)
    print('Done Full Set')


    # In category subset these cause problems
    cleanup.dropFeatures(df, ['installs', 'contentRating'])

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

    models = make_models(category)

    strat = StratifiedKFold(n_splits=10)

    for name, pipe in models:
        cross_val_predict_score(X_all, y_all, pipe, category, name, strat)


def make_models(category) -> list:
    if category is None:
        return best_full_set_models()
    else:
        return best_category_subset_models(category)


def best_category_subset_models(category) -> list:
    p = "cs_"

    # cs\_all \& gradient\_num
    # for category subgroups I just do the one model
    models = []

    if category == "RACING":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.001,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features='sqrt',
            gradient__n_estimators=150
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "PHOTOGRAPHY":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.001,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features=None,
            gradient__n_estimators=100
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "TOOLS":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.001,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features=None,
            gradient__n_estimators=150
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "HEALTH & FITNESS":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.001,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=150
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "MUSIC & AUDIO":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.001,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=150
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "FINANCE":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.001,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=75
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "ARCADE":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.001,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features=None,
            gradient__n_estimators=100
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "STRATEGY":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features='sqrt',
            gradient__n_estimators=100
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "SOCIAL":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features='sqrt',
            gradient__n_estimators=75
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "BUSINESS":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=10
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "WEATHER":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=100
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "ENTERTAINMENT":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=100
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "PUZZLE":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=50
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "ROLE PLAYING":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=75
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "ADVENTURE":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features=None,
            gradient__n_estimators=150
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "PERSONALIZATION":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.01,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features=None,
            gradient__n_estimators=50
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "EDUCATIONAL":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features='sqrt',
            gradient__n_estimators=5
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "MEDICAL":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features='sqrt',
            gradient__n_estimators=5
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "SIMULATION":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features=None,
            gradient__n_estimators=10
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "LIFESTYLE":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features=None,
            gradient__n_estimators=10
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "VIDEO PLAYERS & EDITORS":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features=None,
            gradient__n_estimators=150
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "ACTION":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=5
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "WORD":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=5
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "CASINO":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=5
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "EDUCATION":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=50
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "TRAVEL & LOCAL":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features=None,
            gradient__n_estimators=5
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "PRODUCTIVITY":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features=None,
            gradient__n_estimators=50
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "MAPS & NAVIGATION":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=1,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features=None,
            gradient__n_estimators=10
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "CARD":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=1,
            gradient__loss='deviance',
            gradient__max_depth=3,
            gradient__max_features=None,
            gradient__n_estimators=100
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "CASUAL":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=10
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "BOARD":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=100
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "SPORTS":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=5
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "COMMUNICATION":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features='sqrt',
            gradient__n_estimators=50
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    elif category == "BOOKS & REFERENCE":
        all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category))
        all_gradient_numeric = Pipeline(all_gradient_numeric)
        all_gradient_numeric.set_params(
            gradient__learning_rate=1,
            gradient__loss='deviance',
            gradient__max_depth=5,
            gradient__max_features=None,
            gradient__n_estimators=75
        )
        models.append((p + "all_gradient_numeric", all_gradient_numeric))

    assert (len(models) == 1)
    return models


def best_full_set_models() -> list:
    p = "fd_"
    models = []

    num_forest_numeric, _ = pipes.forest_numeric(category=None)
    num_forest_numeric = Pipeline(num_forest_numeric)
    num_forest_numeric.set_params(
        forest__criterion='gini',
        forest__warm_start=False,
        forest__n_estimators=75,
        forest__max_depth=5,
        forest__max_features=None
    )
    models.append((p + "numeric_forest_numeric", num_forest_numeric))

    num_gradient_numeric, _ = pipes.gradient_numeric(category=None)
    num_gradient_numeric = Pipeline(num_gradient_numeric)
    num_gradient_numeric.set_params(
        gradient__loss='deviance',
        gradient__learning_rate=0.1,
        gradient__n_estimators=150,
        gradient__max_depth=5,
        gradient__max_features='sqrt'
    )
    models.append((p + "numeric_gradient_numeric", num_gradient_numeric))

    all_forest_numeric, _ = pipes.add_text(pipes.forest_numeric(category=None))
    all_forest_numeric = Pipeline(all_forest_numeric)
    all_forest_numeric.set_params(
        forest__criterion='gini',
        forest__warm_start=False,
        forest__n_estimators=50,
        forest__max_depth=3,
        forest__max_features=None
    )
    models.append((p + "all_forest_numeric", all_forest_numeric))

    all_gradient_numeric, _ = pipes.add_text(pipes.gradient_numeric(category=None))
    all_gradient_numeric = Pipeline(all_gradient_numeric)
    all_gradient_numeric.set_params(
        gradient__loss='deviance',
        gradient__learning_rate=1,
        gradient__n_estimators=150,
        gradient__max_depth=3,
        gradient__max_features=None
    )
    models.append((p + "all_gradient_numeric", all_gradient_numeric))

    bnb_text, _ = pipes.bnb_text()
    bnb_text = Pipeline(bnb_text)
    bnb_text.set_params(
        tfidf_vect__analyzer='word',
        tfidf_vect__min_df=0.25,
        tfidf_vect__max_df=0.75,
        tfidf_vect__binary=True,
        tfidf_vect__use_idf=True,
        tfidf_vect__norm='l2',
        bnb__alpha=0.0001,
        bnb__fit_prior=True
    )
    models.append((p + "text_bnb_text", bnb_text))

    gradient_text, _ = pipes.gradient_text()
    gradient_text = Pipeline(gradient_text)
    gradient_text.set_params(
        tfidf_vect__analyzer='word',
        tfidf_vect__min_df=0.25,
        tfidf_vect__max_df=0.6,
        tfidf_vect__binary=False,
        tfidf_vect__use_idf=True,
        tfidf_vect__norm='l2',
        gradient__loss='deviance',
        gradient__learning_rate=0.1,
        gradient__n_estimators=100,
        gradient__max_depth=7,
        gradient__max_features=None
    )
    models.append((p + "text_gradient_text", gradient_text))

    svm_text, _ = pipes.svm_text()
    svm_text = Pipeline(svm_text)
    svm_text.set_params(
        tfidf_vect__analyzer='word',
        tfidf_vect__min_df=0.25,
        tfidf_vect__max_df=0.6,
        tfidf_vect__binary=False,
        tfidf_vect__use_idf=True,
        tfidf_vect__norm='l2',
        svm__C=10,
        svm__kernel='linear',
        svm__gamma=0.01,
        svm__probability=False,
        svm__shrinking=True,
        svm__tol=0.01,
        svm__max_iter=1000
    )
    models.append((p + "text_svm_text", svm_text))

    return models


if __name__ == '__main__':
    main()
