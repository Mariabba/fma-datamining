import itertools
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import missingno as mso

# sklearn
from sklearn import metrics
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    StandardScaler,
    LabelEncoder,
    label_binarize,
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
import seaborn as sns
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
        disp = plot_confusion_matrix(Clf, X, y, cmap="PuRd", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def conf_mat_disp(confusion_matrix, disp_labels):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=disp_labels
    )

    disp.plot(cmap="PuRd")


# DATASET
df = utils.load("../data/tracks.csv", dummies=True, buckets="continuous", fill=True)
column2drop = [
    ("album", "title"),
    ("artist", "name"),
    ("set", "split"),
    ("track", "title"),
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "language_code"),
    ("track", "number"),
    ("track", "tags"),
    ("track", "genres"),  # todo da trattare se si vuole inserire solo lei
    ("track", "genres_all"),
]
df.drop(column2drop, axis=1, inplace=True)
print(df["album", "type"].unique())
print(df["album", "type"].value_counts())
df = df[df["album", "type"] != "Contest"]

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

print(df.info())
print(df["album", "type"].value_counts())
y = mso.matrix(df)
y = plt.show()

print("cosa?", df["album", "type"].unique())

# Create KNN Object.
knn = KNeighborsClassifier(
    n_neighbors=2, p=1
)  # valori migliori dalla gridsearch n = 2, p=1, levarli per avere la standard
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
draw_confusion_matrix
print("Accuracy:", metrics.accuracy_score(y_test, Y_pred))
# confusion matrix
print("\033[1m" "Confusion matrix" "\033[0m")

plot_confusion_matrix(knn, X_test, y_test, cmap="PuRd")
draw_confusion_matrix(knn, X_test, y_test)

print()

print("\033[1m" "Classification report test" "\033[0m")
print(classification_report(y_test, Y_pred))


print()

print("\033[1m" "Metrics" "\033[0m")
print()

print("Accuracy %s" % accuracy_score(y_test, Y_pred))

print("F1-score %s" % f1_score(y_test, Y_pred, labels=[0, 1], average=None))

print("Precision %s" % precision_score(y_test, Y_pred, labels=[0, 1], average=None))

print("Recall %s" % recall_score(y_test, Y_pred, labels=[0, 1], average=None))

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
