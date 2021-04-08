# %matplotlib inline
from pydotplus import graphviz

import utils
from pathlib import Path

# import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import lb

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pydotplus
from sklearn import tree, metrics
from IPython.display import Image
from sklearn.model_selection import StratifiedShuffleSplit

import os
from sklearn.utils import class_weight

os.environ["PATH"] += (
    os.pathsep
    + "C:/Users/saverio/Desktop/Data Mining/DataMiningProject/venvv/Lib/site-packages/graphviz/bin"
)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Decision Tree Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="OrRd", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


def load_data(path):
    df = utils.load_tracks(
        path, outliers=False, buckets="discrete"
    )
    # feature to reshape
    label_encoders = dict()
    column2encode = [
        ("album", "comments"),
        ("album", "favorites"),
        ("album", "listens"),
        ("album", "type"),
        ("artist", "comments"),
        ("artist", "favorites"),
        ("track", "duration"),
        ("track", "comments"),
        ("track", "favorites"),
        ("track", "language_code"),
        ("track", "license"),
        ("track", "listens"),
    ]

    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print(df.info())
    return df


def tuning_param(df, target1, target2):

    # split dataset train and set
    attributes = [col for col in df.columns if col != (target1, target2)]
    X = df[attributes].values
    y = df[target1, target2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25
    )
    print(X_train.shape, X_test.shape)

    # tuning hyperparam with randomize search

    # This has two main benefits over an exhaustive search:

    # A budget can be chosen independent of the number of parameters and possible values.

    # Adding parameters that do not influence the performance does not decrease efficiency.
    # RANDOM
    print("Parameter Tuning: \n")

    # tuning parameters with random search
    print("Search best parameters: \n")
    param_list = {
        "max_depth": [None] + list(np.arange(2, 50)),
        "min_samples_split": [2, 5, 10, 15, 20, 30, 50, 100, 150],
        "min_samples_leaf": [1, 2, 5, 10, 15, 20, 30, 50, 100, 150],
        "criterion": ["gini", "entropy"],
    }

    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1
    )

    random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=100)
    random_search.fit(X, y)
    report(random_search.cv_results_, n_top=3)


def tuning_param_gridsearch(df, target1, target2):

    # split dataset train and set
    attributes = [col for col in df.columns if col != (target1, target2)]
    X = df[attributes].values
    y = df[target1, target2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20
    )
    print(X_train.shape, X_test.shape)
    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1
    )

    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results["rank_test_score"] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print(
                    "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results["mean_test_score"][candidate],
                        results["std_test_score"][candidate],
                    )
                )
                print("Parameters: {0}".format(results["params"][candidate]))
                print("")

    param_list = {
        "max_depth": [None] + list(np.arange(2, 50)),
        "min_samples_split": list(np.arange(2, 50)),
        "min_samples_leaf": list(np.arange(2, 50)),
        "criterion": ["gini", "entropy"],
    }

    grid_search = GridSearchCV(clf, param_grid=param_list)
    grid_search.fit(X, y)
    clf = grid_search.best_estimator_
    print(report(grid_search.cv_results_, n_top=3))


def build_model(
    df, target1, target2, min_samples_split, min_samples_leaf, max_depth, criterion
):

    # split dataset train and set
    attributes = [col for col in df.columns if col != (target1, target2)]
    X = df[attributes].values
    y = df[target1, target2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

    print(X_train.shape, X_test.shape)
    # build a model
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        #class_weight={0:1, 1:4, 2:3, 3:5}
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # value importance
    for col, imp in zip(attributes, clf.feature_importances_):
        print(col, imp)

    # Apply the decision tree on the training set
    print("Apply the decision tree on the training set: \n")
    y_pred = clf.predict(X_train)
    print("Accuracy %s" % accuracy_score(y_train, y_pred))
    print("F1-score %s" % f1_score(y_train, y_pred, average=None))

    print(classification_report(y_train, y_pred))

    confusion_matrix(y_train, y_pred)

    # Apply the decision tree on the test set and evaluate the performance
    print("Apply the decision tree on the test set and evaluate the performance: \n")
    y_pred = clf.predict(X_test)

    print("Accuracy %s" % accuracy_score(y_test, y_pred))
    print("F1-score %s" % f1_score(y_test, y_pred, average=None))
    print(classification_report(y_test, y_pred))
    confusion_matrix(y_test, y_pred)

    # ROC Curve
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    lb.fit(y_test)
    lb.classes_.tolist()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    by_test = lb.transform(y_test)
    by_pred = lb.transform(y_pred)
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(by_test[:, i], by_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    roc_auc = roc_auc_score(by_test, by_pred, average=None)
    print(roc_auc)

    plt.figure(figsize=(8, 5))
    for i in range(4):
        plt.plot(
            fpr[i],
            tpr[i],
            label="%s ROC curve (area = %0.2f)" % (lb.classes_.tolist()[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=22)
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.show()

    # Model Accuracy, how often is the classifier correct?
    draw_confusion_matrix
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # confusion matrix
    print("\033[1m" "Confusion matrix" "\033[0m")

    draw_confusion_matrix(clf, X_test, y_test)

    print()

    for col, imp in zip(attributes, clf.feature_importances_):
        print(col, imp)

    top_n = 10
    feat_imp = pd.DataFrame(columns=["columns", "importance"])
    for col, imp in zip(attributes, clf.feature_importances_):
        feat_imp = feat_imp.append(
            {"columns": col, "importance": imp}, ignore_index=True
        )
    print(feat_imp)

    feat_imp.sort_values(by="importance", ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]

    feat_imp.plot(
        title="Top 10 features contribution",
        x="columns",
        fontsize=8.5,
        rot=15,
        y="importance",
        kind="bar",
        colormap="Pastel1",
    )
    plt.show()




tracks = load_data("data/tracks.csv")
# tuning_param(tracks, "album", "type")
# tuning_param_gridsearch(tracks, "album", "type")
build_model(tracks, "album", "type", 100, 100, 8, "entropy")
#build_model(tracks, "album", "type", 2, 1, 20, "entropy")
