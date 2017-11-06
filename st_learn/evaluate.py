import inspect
import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Statics
FILE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
RESOURCES_DIR = os.path.join(FILE_DIR, '../resources/')


def class_name(obj):
    cn = str(obj.__class__)
    cn = re.search("([A-Z])\w+", cn).group(0)
    return cn


def handleEstimator(rs, p, p_val, cnt):
    rs['step_' + cnt + '_name'] = p
    rs['step_' + cnt + '_est'] = class_name(p_val)


def handleFeatures(rs, p, p_val):
    d = {k: v for (k, v) in p_val if v is not None}
    for k, v in d.items():
        rs[p + '_' + k] = str(list(map(lambda x: class_name(x), v)))


def onSearch(rs, search, scoring, STRAT_SPLITS):

    # document the grid
    try:
        rs['param_distributions'] = str(search.param_distributions)
    except Exception:
        pass

    try:
        rs['param_grid'] = str(search.param_grid)
    except Exception:
        pass

    rs['best_params_'] = str(search.best_params_)
    rs['folds'] = str(STRAT_SPLITS)


    # document the best estimators steps with params
    pipe = search.best_estimator_

    names = [name for (name, est) in pipe.steps]
    params = pipe.get_params(deep=True)

    cnt = 1
    for name in names:
        this_params = {k for k, v in params.items() if k.startswith(name)}

        for p in this_params:
            p_val = params[p]
            if p == name:
                # document which steps have been executed
                handleEstimator(rs, p, p_val, str(cnt))
            elif p.endswith('__features'):
                handleFeatures(rs, p, p_val)
            else:
                # and with which settings
                rs[p] = str(params[p])

        cnt += 1


    # Document scores
    cv_results = search.cv_results_

    for scorer in scoring:
        rs['score_' + scorer + '_train_mean'] = cv_results['mean_train_' + scorer][search.best_index_]
        rs['score_' + scorer + '_train_std'] = cv_results['std_train_' + scorer][search.best_index_]
        rs['score_' + scorer + '_test_mean'] = cv_results['mean_test_' + scorer][search.best_index_]
        rs['score_' + scorer + '_test_std'] = cv_results['std_test_' + scorer][search.best_index_]

    rs['time_fit_mean'] = cv_results['mean_fit_time'][search.best_index_]
    rs['time_fit_std'] = cv_results['std_fit_time'][search.best_index_]
    rs['time_score_mean'] = cv_results['mean_score_time'][search.best_index_]
    rs['time_score_std'] = cv_results['std_score_time'][search.best_index_]

    return rs


def printToCsv(rs, model):
    csv = os.path.join(RESOURCES_DIR, 'evaluation_' + model + '.csv')

    if os.path.exists(csv):
        df = pd.read_csv(csv, error_bad_lines=False, index_col=0)
        df = df.append(rs, ignore_index=True)
    else:
        df = pd.DataFrame().append(rs, ignore_index=True)

    df.index.name = "Runs"
    df.to_csv(path_or_buf=csv, index=True)


def printToConsole(rs):
    print("\n")
    print("{:}={:.3} (+/- {:.3})"
          .format('score_precision_eval_mean', rs.score_precision_eval_mean, rs.score_precision_eval_std))
    print("{:}={:.3} (+/- {:.3})"
          .format('score_recall_eval_mean', rs.score_recall_eval_mean, rs.score_recall_eval_std))
    print("{:}={:.3} (+/- {:.3})"
          .format('score_f1_eval_mean', rs.score_f1_eval_mean, rs.score_f1_eval_std))
    print("{:}={:.3}"
          .format('score_accuracy_eval', rs.score_accuracy_eval))


def onMetrics(rs, y_eval, y_pred):
    # cannot do ROC AUC with multi-label
    prc, rec, f1, sup = precision_recall_fscore_support(y_eval, y_pred, beta=1)
    rs['score_precision_eval_mean'] = prc.mean()
    rs['score_precision_eval_std'] = prc.std()
    rs['score_recall_eval_mean'] = rec.mean()
    rs['score_recall_eval_std'] = rec.std()
    rs['score_f1_eval_mean'] = f1.mean()
    rs['score_f1_eval_std'] = f1.std()
    rs['score_accuracy_eval'] = accuracy_score(y_eval, y_pred)
    rs['score_kappa_eval'] = cohen_kappa_score(y_eval, y_pred)


def conf_mat(y_true, y_pred, labels, classes, normalize=True, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    # measures

    # acc = accuracy_score(y_true, y_pred, normalize)
    # print('ACC: ' + str(np.round(acc, 3)))

    # How much better than always predicting the most frequent class
    null_acc_gain = ((y_true.value_counts()[0] / len(y_true)) / accuracy_score(y_true, y_pred, normalize)) - 1
    print(str(np.round(null_acc_gain, 3)) + ' = null-accuracy gain')


    # Specificity - How "specific" (or "selective") is the classifier in predicting positive instances?
    specificity = tn / (tn + fp)
    print(str(np.round(specificity, 3)) + ' = specificity')


    print(classification_report(y_true, y_pred, digits=3))

    # histogram of predicted probabilities


    # plot - Confusion matrix
    title = 'Confusion matrix'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title += ', normalized'

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # TODO: do we show always?
    plt.show()


# Trade-off sensitivity and specificity
def roc(y_true, y_proba_pos_dict):

    # flushes any previous results
    plt.figure()

    # calculate the auc score early to sort later
    temp = {}
    for name, array in y_proba_pos_dict.items():
        auc_score = roc_auc_score(y_true, array)
        temp[auc_score] = (name, array)


    colors = np.array(['b', 'g', 'm', 'c', 'y', 'r', 'k'])

    counter = 0
    for auc_score in sorted(temp):

        name = temp[auc_score][0]
        array = temp[auc_score][1]

        fpr, tpr, thresholds = roc_curve(y_true, array)

        c = colors[counter % len(colors)]

        plt.plot(fpr, tpr, color=c, lw=1, label='(auc = %0.3f) ' % auc_score + name)
        counter += 1


    # Line for Random Guess
    plt.plot([0, 1], [0, 1], color='lightgrey', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    # Plot settings
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')



# TODO: Gain chart
# TODO: Lift chart
# TODO: variable statistics and plots
# TODO: anything with graphviz?



# great plotting resources
# http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/
# from mlxtend.evaluate import confusion_matrix
# from mlxtend.plotting import plot_confusion_matrix
# cm = confusion_matrix(y_eval, y_pred, binary=False)
# fig, ax = plot_confusion_matrix(conf_mat=cm)
# plt.show()
# print(cm / len(X_train))



# report_roc(y_all_true, {'y_all_proba_pos': y_all_proba_pos})
# plt.show()


# from matplotlib.colors import ListedColormap
# import matplotlib.pyplot as plt
#
#
# def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#
#     # setup marker generator and color map
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#
#     # plot the decision surface
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0],
#                     y=X[y == cl, 1],
#                     alpha=0.8,
#                     c=colors[idx],
#                     marker=markers[idx],
#                     label=cl,
#                     edgecolor='black')
#
#     # highlight test samples
#     if test_idx:
#         # plot all samples
#         X_test, y_test = X[test_idx, :], y[test_idx]
#
#         plt.scatter(X_test[:, 0],
#                     X_test[:, 1],
#                     c='',
#                     edgecolor='black',
#                     alpha=1.0,
#                     linewidth=1,
#                     marker='o',
#                     s=100,
#                     label='test set')
#
#
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
#
# plot_decision_regions(X=X_combined_std, y=y_combined,
#                                             classifier=ppn, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
#
# plt.tight_layout()
# #plt.savefig('images/03_01.png', dpi=300)
# plt.show()
