import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import prepare


# adding a stemmer: http://nlpforhackers.io/text-classification/
def stemming_tokenizer(text):
    ps = PorterStemmer()
    return [ps.stem(w) for w in word_tokenize(text.replace('\\n', ' '))]


def only_text() -> (list, dict):
    # in-sync with report
    return \
        [
            # pipe steps
            ('only_text', FunctionTransformer(prepare.only_text, validate=False)),
            # ('only_text', prepare.ColumnSelector(['text'])),
            ('tfidf_vect', TfidfVectorizer(
                lowercase=True,
                tokenizer=stemming_tokenizer,
                stop_words=stopwords.words('english') + list(string.punctuation)
            ))
        ], {
            # param settings
            "tfidf_vect__analyzer": ['word'],
            # we see better results with smaller vocab
            "tfidf_vect__min_df": [0.25, 0.4],
            "tfidf_vect__max_df": [0.6, 0.75],
            "tfidf_vect__binary": [False],
            "tfidf_vect__use_idf": [True],
            "tfidf_vect__norm": ['l2']
        }


def bnb_text() -> (list, dict):
    # in-sync with report
    text_steps, text_params = only_text()
    pipe_steps = []
    pipe_steps.extend(text_steps)
    pipe_steps.extend([
        ('bnb', BernoulliNB())
    ])
    pipe_params = {}
    pipe_params.update(text_params)
    pipe_params.update({
        # bernoulli works well with binary and shorter texts
        'tfidf_vect__binary': [True],
        'bnb__alpha': [0.0001, 0.001, 0.01],
        'bnb__fit_prior': [True]
    })

    # BernoulliNB b/c
    # http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

    return pipe_steps, pipe_params


def gradient_text() -> (list, dict):
    # in-sync with report
    text_steps, text_params = only_text()
    pipe_steps = []
    pipe_steps.extend(text_steps)
    pipe_steps.extend([
        ('gradient', GradientBoostingClassifier())
    ])
    pipe_params = {}
    pipe_params.update(text_params)
    pipe_params.update({
        "gradient__loss": ['deviance'],
        "gradient__learning_rate": [0.1, 1],
        "gradient__n_estimators": [50, 100],
        "gradient__max_depth": [5, 7],
        "gradient__max_features": [None]
    })

    return pipe_steps, pipe_params


def svm_text() -> (list, dict):
    # in-sync with report
    text_steps, text_params = only_text()
    pipe_steps = []
    pipe_steps.extend(text_steps)
    pipe_steps.extend([
        ('svm', SVC())
    ])
    pipe_params = {}
    pipe_params.update(text_params)
    pipe_params.update({
        "svm__C": [1, 10, 100],
        "svm__kernel": ['linear'],
        "svm__gamma": [0.01, 0.1, 1],
        "svm__probability": [False],
        "svm__shrinking": [True],
        "svm__tol": [0.01],
        "svm__max_iter": [1000]
    })

    # https://stats.stackexchange.com/questions/37669/libsvm-reaching-max-number-of-iterations-warning-and-cross-validation

    return pipe_steps, pipe_params


def forest_numeric(category) -> (list, dict):
    # in-sync with report
    pipe_steps = [
        ('drop_text', FunctionTransformer(prepare.drop_text, validate=False)),
        ('encode', prepare.categoricals_encode(category)),
        ('forest', RandomForestClassifier())
    ]
    pipe_params = {}
    pipe_params.update({
        "forest__criterion": ['gini'],
        # "forest__warm_start": [True, False],
        "forest__warm_start": [False],
        # "forest__n_estimators": [5, 10, 50, 75, 100, 150],
        "forest__n_estimators": [50, 75, 100],
        "forest__max_depth": [3, 5],
        # "forest__max_features": ['sqrt', None]
        "forest__max_features": [None]
    })

    return pipe_steps, pipe_params


def gradient_numeric(category) -> (list, dict):
    # in-sync with report
    pipe_steps = [
        ('drop_text', FunctionTransformer(prepare.drop_text, validate=False)),
        ('encode', prepare.categoricals_encode(category)),
        ('gradient', GradientBoostingClassifier())
    ]
    pipe_params = {}
    pipe_params.update({
        "gradient__loss": ['deviance'],
        # "gradient__learning_rate": [0.001, 0.01, 0.1, 1, 10],
        "gradient__learning_rate": [0.01, 0.1, 1],
        # "gradient__n_estimators": [5, 10, 50, 75, 100, 150],
        "gradient__n_estimators": [150],
        "gradient__max_depth": [3, 5],
        # "gradient__max_features": ['sqrt', None]
        "gradient__max_features": [None]
    })

    return pipe_steps, pipe_params


# generally used for debugging other parts
def quick_numeric(category) -> (list, dict):
    pipe_steps = [
        ('drop_text', FunctionTransformer(prepare.drop_text, validate=False)),
        ('encode', prepare.categoricals_encode(category)),
        ('tree', DecisionTreeClassifier())
    ]
    pipe_params = {}
    pipe_params.update({
        "tree__max_depth": [3],
        "tree__max_features": ['sqrt']
    })

    return pipe_steps, pipe_params


def dummy(category) -> (list, dict):
    pipe_steps = [
        ('drop_text', FunctionTransformer(prepare.drop_text, validate=False)),
        ('encode', prepare.categoricals_encode(category)),
        ('dummy', DummyClassifier())
    ]

    return pipe_steps, {}


def add_text(steps_params: (list, dict)) -> (list, dict):
    steps = [("text_transform", TextFeatureTransformer())]
    steps.extend(steps_params[0])
    return steps, steps_params[1]


class TextFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        # 0.373 f1_weighted_test_mean on fd_text
        (steps, params) = gradient_text()

        self.estimator = Pipeline(steps)

        # set best params for estimator
        self.estimator.set_params(
            tfidf_vect__analyzer='word',
            tfidf_vect__binary=False,
            tfidf_vect__min_df=0.25,
            tfidf_vect__max_df=0.6,
            tfidf_vect__norm='l2',
            tfidf_vect__use_idf=True,
            gradient__learning_rate=0.1,
            gradient__loss='deviance',
            gradient__max_depth=7,
            gradient__max_features=None,
            gradient__n_estimators=100
        )

    def fit(self, X, y=None):
        # fit the estimator using copies (just to be safe)
        self.estimator.fit(X.copy(), y.copy())

        return self

    def transform(self, X):
        X_ = X.copy()

        X_['textFeaturePrediction'] = self.estimator.predict(X_)

        return X_
