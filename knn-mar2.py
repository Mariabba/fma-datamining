import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# sklearn
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    StandardScaler,
    LabelEncoder,
)
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    make_scorer,
    precision_recall_curve,
)
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier

from pandas import DataFrame

from pandas import DataFrame
import utils
from pathlib import Path


# FUNCTION
def plot_grid_search_results(grid_results, grid_best_params, grid_best_score):
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(12, 18)
    )  # figsize = (13,30)

    ax1.errorbar(
        x=grid_results["param_n_neighbors"].data,
        y=grid_results["mean_test_f1"],
        color="#B8002E",
        yerr=grid_results["std_test_f1"],
        ecolor="orange",
    )
    ax1.set_xlabel("n_neigbors")
    ax1.set_ylabel("f1")
    ax1.set_title("Classification F1")

    ax2.errorbar(
        x=grid_results["param_n_neighbors"].data,
        y=grid_results["mean_test_recall"],
        color="#B8002E",
        yerr=grid_results["std_test_recall"],
        ecolor="orange",
    )
    ax2.set_xlabel("n_neigbors")
    ax2.set_ylabel("recall")
    ax2.set_title("Classification Recall")

    ax3.errorbar(
        x=grid_results["param_n_neighbors"].data,
        y=grid_results["mean_test_roc_auc"],
        color="#B8002E",
        yerr=grid_results["std_test_roc_auc"],
        ecolor="orange",
    )
    ax3.plot(
        [grid_best_params["n_neighbors"]],
        [grid_best_score],
        marker=".",
        markeredgewidth=3,
        c="r",
    )
    ax3.annotate(
        "best k",
        xy=(grid_best_params["n_neighbors"], grid_best_score),
        xytext=(grid_best_params["n_neighbors"] + 1, grid_best_score + 0.02),
        arrowprops=dict(facecolor="black", shrink=0.01),
    )
    ax3.set_xlabel("n_neigbors")
    ax3.set_ylabel("roc auc")
    ax3.set_title("Classification Roc Auc")

    fig.tight_layout()
    plt.show()


def results_permutation_importance(res, attr):
    # get importance
    importance = res.importances_mean

    # summarize feature importance
    feature_importances = []
    for col, imp in zip(attr, importance):
        feature_importances.append((col, imp))

    sorted_feature_importances = sorted(
        feature_importances, key=lambda tup: (-tup[1], tup[0])
    )

    return sorted_feature_importances


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="OrRd", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def conf_mat_disp(confusion_matrix, disp_labels):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=disp_labels
    )

    disp.plot(cmap="OrRd")


def draw_roc_curve(Y_test, Y_pred, diz, k):
    fig, ax = plt.subplots()  # figsize = (13,30)

    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(Y_test, Y_pred, average=None)

    diz[k] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc": roc_auc}

    ax.plot(fpr, tpr, color="#994D00", label="ROC curve (area = %0.2f)" % (roc_auc))
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Roc curve of the model")
    ax.tick_params(axis="both")
    ax.legend(loc="lower right", title="AUC", fontsize=14, frameon=True)

    fig.tight_layout()
    plt.show()


def draw_precision_recall_curve(Y_test, Y_pred):
    fig, ax = plt.subplots()

    pr_ap = average_precision_score(Y_test, Y_pred, average=None)
    precision, recall, ap_thresholds = precision_recall_curve(Y_test, Y_pred)

    ax.plot(precision, recall, color="#994D00", label="AP %0.4f" % (pr_ap))
    # ax.plot([0, 1], [no_skill, no_skill], 'r--', label='%0.4f' % no_skill)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.tick_params(axis="both")
    ax.legend(loc="upper right", title="AP", frameon=True)
    ax.set_title("Model Precision-Recall curve")

    fig.tight_layout()
    plt.show()


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


# DATASET
df = utils.load(Path("data/tracks.csv"), clean=True, dummies=True)
column2drop = [
    ("album", "title"),  # add later
    ("artist", "name"),  # add later
    ("set", "split"),
    ("track", "title"),
    ("album", "date_created"),
    ("artist", "date_created"),
    ("track", "date_created"),
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "tags"),
    ("track", "genres"),
    ("track", "genres_all"),
]
df.drop(column2drop, axis=1, inplace=True)

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


# Create KNN Object. #DA FINIRE MARI
knn = KNeighborsClassifier()
# Create x and y variables.
x = df.drop(columns=[("album", "type")])
y = df[("album", "type")]
# Split data into training and testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
# Training the model.
knn.fit(x_train, y_train)
# Predict test data set.
# y_pred = knn.predict(x_test)
# Checking performance our model with classification report.
# print(classification_report(y_test, y_pred))
# Checking performance our model with ROC Score.
# roc_auc_score(y_test, y_pred)

# tuning
# List Hyperparameters that we want to tune.
n_neighbors = list(range(1, 10))
p = [1, 2]
# Convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p)
# Create new KNN object
knn_2 = KNeighborsClassifier()
# Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
# Fit the model
best_model = clf.fit(x, y)
# Print The value of best Hyperparameters
print("Best p:", best_model.best_estimator_.get_params()["p"])
print("Best n_neighbors:", best_model.best_estimator_.get_params()["n_neighbors"])
