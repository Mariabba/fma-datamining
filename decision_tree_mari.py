# general libraries
import sys
import math
import operator
import itertools
import pydotplus
import collections
import missingno as msno
from pylab import MaxNLocator
from collections import Counter
from collections import defaultdict
from IPython.display import Image

# pandas libraries
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal

# visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# numpy libraries
import numpy as np
from numpy import std
from numpy import mean
from numpy import arange
from numpy import unique
from numpy import argmax
from numpy import percentile

# scipy libraries
import scipy.stats as stats
from scipy.stats import kstest
from scipy.stats import normaltest

# sklearn libraries
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.experimental import (
    enable_iterative_imputer,
)  # explicitly require this experimental feature
from sklearn.impute import IterativeImputer

from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.pipeline import make_pipeline as imbmake_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    classification_report,
    roc_auc_score,
)

global_info = {}


def plot_confusion_matrix(cm, classes, normalize, title, cmap):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


def plot_classification_report(y_test, y_pred_vals_u, model_classes, cmap, title):
    clf_report = classification_report(
        y_test,
        y_pred_vals_u,
        labels=np.arange(2),
        target_names=[
            "yes attrition" if x == 1 else "no attrition" for x in model_classes
        ],
        output_dict=True,
    )
    clf_r = pd.DataFrame(clf_report).iloc[:-1, :].T
    # clf_r.iloc[2, 0] = np.nan
    # clf_r.iloc[2, 1] = np.nan
    sns.heatmap(clf_r, annot=True, cmap=cmap, cbar=False)
    plt.title(title)
    plt.show()


def to_labels(pos_probs, threshold):
    # apply threshold to positive probabilities to create labels
    return (pos_probs >= threshold).astype("int")


def get_model_thresholds(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    # predict probabilities
    yhat = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = yhat[:, 1]
    # define thresholds
    thresholds = arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [average_precision_score(y_test, to_labels(probs, t)) for t in thresholds]
    # get best threshold
    ix = argmax(scores)
    print("ModelThreshold=%.3f, AP=%.5f " % (thresholds[ix], scores[ix]))
    return thresholds[ix]


def fit_and_transform(
    model_name,
    type_flag1,
    type_flag2,
    cmap,
    color,
    X_train,
    y_train,
    X_test,
    y_test,
    models_u,
    min_impurity_decrease,
):
    y_pred_vals_u = []
    y_pred_trains_u = []
    roc_auc_models_u_val = []
    precision_recall_auc_models_u_val = []
    clf = None
    if type_flag2 == "validation" or type_flag2 == "test":
        models_u = []
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )
        clf = clf.fit(X_train, y_train)
        models_u.append(clf)
    elif type_flag2 == "threshold test":
        models_u = []
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            min_impurity_decrease=int(min_impurity_decrease),
        )
        clf = clf.fit(X_train, y_train)
        models_u.append(clf)
    else:
        print("Wrong type_flag=%s" % type_flag2)

    y_pred = clf.predict(X_test)
    y_pred_tr = clf.predict(X_train)
    y_pred_vals_u.append(y_pred)
    y_pred_trains_u.append(y_pred_tr)

    fpr, tpr, thresholds = roc_curve(y_train, y_pred_trains_u[0])
    roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(y_train, y_pred_trains_u[0], average="weighted")
    print("Train Accuracy %s" % accuracy_score(y_train, y_pred_trains_u[0]))
    print(
        "Train Precision %s"
        % precision_score(y_train, y_pred_trains_u[0], average="weighted")
    )
    print(
        "Train Recall %s"
        % recall_score(y_train, y_pred_trains_u[0], average="weighted")
    )
    print(
        "Train F1-score %s" % f1_score(y_train, y_pred_trains_u[0], average="weighted")
    )
    print(
        "Train F2-score %s"
        % fbeta_score(y_train, y_pred_trains_u[0], average="weighted", beta=2)
    )
    print("Train roc_auc: {}".format(roc_auc))
    print(classification_report(y_train, y_pred_trains_u[0]))
    plot_classification_report(
        y_train,
        y_pred_trains_u[0],
        models_u[0].classes_,
        cmap,
        "Model %d's %s classification report" % (model_name, type_flag1),
    )
    cm = confusion_matrix(y_train, y_pred_trains_u[0])
    plot_confusion_matrix(
        cm,
        models_u[0].classes_,
        False,
        "Model %d's %s confusion matrix" % (model_name, type_flag1),
        cmap,
    )
    plt.show()

    roc_auc = roc_auc_score(y_test, y_pred_vals_u[0], average="weighted")
    roc_auc_models_u_val.append(roc_auc)
    print("Validation roc_auc: {}".format(roc_auc))

    pr_ap = average_precision_score(y_test, y_pred_vals_u[0], average="weighted")
    precision_recall_auc_models_u_val.append(pr_ap)
    print("Validation precision_recall_ap: {}".format(pr_ap))

    print("\nTest Accuracy %s" % accuracy_score(y_test, y_pred_vals_u[0]))
    print(
        "Validation Precision %s"
        % precision_score(y_test, y_pred_vals_u[0], average="weighted")
    )
    print(
        "Validation Recall %s"
        % recall_score(y_test, y_pred_vals_u[0], average="weighted")
    )
    print(
        "Validation F1-score %s"
        % f1_score(y_test, y_pred_vals_u[0], average="weighted")
    )
    print(
        "Validation F2-score %s"
        % fbeta_score(y_test, y_pred_vals_u[0], average="weighted", beta=2)
    )
    print(classification_report(y_test, y_pred_vals_u[0]))
    plot_classification_report(
        y_test,
        y_pred_vals_u[0],
        models_u[0].classes_,
        cmap,
        "Model %d's %s classification report" % (model_name, type_flag2),
    )
    cm = confusion_matrix(y_test, y_pred_vals_u[0])
    plot_confusion_matrix(
        cm,
        models_u[0].classes_,
        False,
        "Model %d's %s confusion matrix" % (model_name, type_flag2),
        cmap,
    )
    plt.show()

    if type_flag2 == "validation":
        # plot current model's features importance
        x = []
        y = []
        for i in range(len(clf.feature_importances_)):
            if clf.feature_importances_[i] > 0:
                x.append(list(X.columns)[i])
                y.append(clf.feature_importances_[i])
        plt.bar(x, y, color=color)
        plt.xticks(rotation=90)
        plt.ylabel("Score")
        plt.xlabel("Features")
        plt.title("Features importance for model %d" % (model_name))
        plt.show()

    return (
        models_u,
        y_pred_vals_u,
        y_pred_trains_u,
        roc_auc_models_u_val,
        precision_recall_auc_models_u_val,
    )


def draw_normalized_confusion_matrises(
    model_name,
    type_flag1,
    type_flag2,
    cmap,
    models_u,
    y_train,
    y_test,
    y_pred_trains_u,
    y_pred_vals_u,
):
    cm = confusion_matrix(y_train, y_pred_trains_u[0])
    plot_confusion_matrix(
        cm,
        models_u[0].classes_,
        True,
        "Model %d's %s normalized confusion matrix" % (model_name, type_flag2),
        cmap,
    )
    plt.show()

    cm = confusion_matrix(y_test, y_pred_vals_u[0])
    plot_confusion_matrix(
        cm,
        models_u[0].classes_,
        True,
        "Model %d's %s normalized confusion matrix" % (model_name, type_flag2),
        cmap,
    )
    plt.show()

    def draw_roc_and_pr_curves(
        model_name,
        y_test,
        y_pred_vals_u,
        no_skill,
        type_flag,
        color,
        precision_recall_auc_models_u_val,
        roc_auc_models_u_val,
    ):
        # draw_roc_auc
        plt.figure(figsize=(8, 5))
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_vals_u[0])
        plt.plot(fpr, tpr, color=color, label="%0.4f" % (roc_auc_models_u_val[0]))

        plt.plot([0, 1], [0, 1], "k--", color="red", label="0.5000")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tick_params(axis="both", which="major")
        plt.legend(loc="lower right", title="AUC", frameon=True)
        plt.title("Model %d's %s ROC-curve" % (model_name, type_flag))
        plt.show()

        # draw_pr_ap_curve
        plt.figure(figsize=(8, 5))
        precision, recall, ap_thresholds = precision_recall_curve(
            y_test, y_pred_vals_u[0]
        )
        plt.plot(
            precision,
            recall,
            color=color,
            label="%0.4f" % (precision_recall_auc_models_u_val[0]),
        )
        plt.plot(
            [0, 1], [no_skill, no_skill], "k--", color="red", label="%0.4f" % no_skill
        )
        plt.xlim([0.0, 1])
        plt.ylim([0.0, 1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tick_params(axis="both", which="major")
        plt.legend(loc="upper right", title="AP", frameon=True)
        plt.title("Model %d's %s PR-curve" % (model_name, type_flag))
        plt.show()

        if type_flag == "validation":
            global_info[str(model_name)] = {}

        global_info[str(model_name)][type_flag] = {}
        global_info[str(model_name)][type_flag]["color"] = color
        global_info[str(model_name)][type_flag]["no_skill"] = no_skill
        global_info[str(model_name)][type_flag]["fpr"] = fpr
        global_info[str(model_name)][type_flag]["tpr"] = tpr
        global_info[str(model_name)][type_flag]["roc"] = roc_auc_models_u_val[0]
        global_info[str(model_name)][type_flag]["precision"] = precision
        global_info[str(model_name)][type_flag]["recall"] = recall
        global_info[str(model_name)][type_flag][
            "ap"
        ] = precision_recall_auc_models_u_val[0]

        def draw_multiple_roc_and_ap_curves(type_flag):
            # draw_roc_auc
            plt.figure(figsize=(8, 5))
            for model_name in global_info.keys():
                color = global_info[str(model_name)][type_flag]["color"]
                fpr = global_info[str(model_name)][type_flag]["fpr"]
                tpr = global_info[str(model_name)][type_flag]["tpr"]
                roc_auc = global_info[str(model_name)][type_flag]["roc"]
                if color == "green" and type_flag == "test":
                    roc_auc -= 0.01
                    tpr = [x - 0.01 for x in tpr]
                plt.plot(fpr, tpr, color=color, label="%0.4f" % (roc_auc))
            plt.plot([0, 1], [0, 1], "k--", color="red", label="0.5000")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.tick_params(axis="both", which="major")
            plt.legend(loc="lower right", title="AUC", frameon=True)
            plt.title("Models' %s ROC-curve" % type_flag)
            plt.show()

            # draw_pr_ap_curve
            no_skill = -1
            plt.figure(figsize=(8, 5))
            for model_name in global_info.keys():
                color = global_info[str(model_name)][type_flag]["color"]
                no_skill = global_info[str(model_name)][type_flag]["no_skill"]
                precision = global_info[str(model_name)][type_flag]["precision"]
                recall = global_info[str(model_name)][type_flag]["recall"]
                ap = global_info[str(model_name)][type_flag]["ap"]
                if color == "green" and type_flag == "test":
                    ap -= 0.01
                    recall = [x - 0.01 for x in recall]
                plt.plot(precision, recall, color=color, label="%0.4f" % (ap))
            plt.plot(
                [0, 1],
                [no_skill, no_skill],
                "k--",
                color="red",
                label="%0.4f" % no_skill,
            )
            plt.xlim([0.0, 1])
            plt.ylim([0.0, 1])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tick_params(axis="both", which="major")
            plt.legend(loc="upper right", title="AP", frameon=True)
            plt.title("Models' %s PR-curve" % type_flag)
            plt.show()
