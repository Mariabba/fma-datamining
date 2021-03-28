import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# sklearn
from sklearn import metrics
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


# DATASET
df = utils.load("../data/tracks.csv", dummies=True, buckets="continuous", fill=True)
column2drop = [
    ("album", "title"),
    ("album", "type"),
    ("album", "producer"),
    ("set", "split"),
    ("track", "title"),
    ("album", "tags"),
    ("album", "id"),
    ("artist", "website"),
    ("artist", "name"),
    ("artist", "tags"),
    ("artist", "wikipedia_page"),
    ("artist", "bio"),
    ("artist", "id"),
    ("track", "language_code"),
    ("track", "composer"),
    ("track", "information"),
    ("track", "license"),
    ("track", "tags"),
    ("track", "genres"),
    ("track", "genres_all"),
]
df.drop(column2drop, axis=1, inplace=True)
print(df.info())


# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "listens"),
    ("album", "type"),
    ("track", "license"),
]
for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

"""
# Create KNN Object.
knn = KNeighborsClassifier(
    n_neighbors=2, p=1
)  # dopo la grid search il mio setting è n = 2 e p =1
# Create x and y variables.
x = df.drop(columns=[("album", "type")])
y = df[("album", "type")]
# Split data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)
print(X_train.shape, X_test.shape)
# Training the model.
knn.fit(X_train, y_train)
# Predict test data set.
Y_pred = knn.predict(X_test)


# Checking performance our model with classification report.

print("Accuracy:", metrics.accuracy_score(y_test, Y_pred))
# confusion matrix
print("\033[1m" "Confusion matrix" "\033[0m")


draw_confusion_matrix(knn, X_test, y_test)

print()


print("\033[1m" "Classification report test" "\033[0m")
print(classification_report(y_test, Y_pred))

print()

print("\033[1m" "Metrics" "\033[0m")


print("Accuracy %s" % accuracy_score(y_test, Y_pred))

print("F1-score %s" % f1_score(y_test, Y_pred, average="weighted", zero_division=0))

print(
    "Precision %s"
    % precision_score(y_test, Y_pred, average="weighted", zero_division=0)
)

print("Recall %s" % recall_score(y_test, Y_pred, average="weighted", zero_division=0))

"""
"""
# TODO TESTARE I PARAMENTRI MIGLIORI
# List Hyperparameters that we want to tune.
print("STA FACENDO LA GRIDSEARCH")
n_neighbors = list(range(1, 10))
p = [1, 2]
# Convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p)
# Create new KNN object
knn_2 = KNeighborsClassifier()
# Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters)
# Fit the model
best_model = clf.fit(x, y)
# Print The value of best Hyperparameters
print("Best p:", best_model.best_estimator_.get_params()["p"])
print("Best n_neighbors:", best_model.best_estimator_.get_params()["n_neighbors"])
"""
