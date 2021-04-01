import itertools
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.colors import ListedColormap
import missingno as mso
from collections import Counter
from collections import defaultdict

# sklearn
from sklearn import metrics
from sklearn.decomposition import PCA
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
df = utils.load(
    "data/tracks.csv",
    dummies=True,
    buckets="continuous",
    fill=True,
    outliers=True,
)

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
    "Unnamed: 0",
    "0",
]
df.drop(column2drop, axis=1, inplace=True)
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

# Create KNN Object CLASSIC
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
print("F1-score %s" % f1_score(y_test, Y_pred, average=None))
print("Precision %s" % precision_score(y_test, Y_pred, average=None))
print("Recall %s" % recall_score(y_test, Y_pred, average=None))

"""EMBALANCE LEARNING"""
# PRINTING PCA FOR COMPARISON

print("Train shape")  # , X_train.shape())
pca = PCA(n_components=4)
pca.fit(X_train)
X_pca = pca.transform(X_train)
X_pca.shape

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_train,
    cmap="Paired",
    edgecolor="k",
    alpha=0.5,
)
plt.title("Standard KNN-PCA")
plt.show()


"""UNDERSAMPLING-RANDOM UNDERSAMPLING"""
rus = RandomUnderSampler(random_state=42, replacement=True)
X_res, y_res = rus.fit_resample(X_train, y_train)
print("Original dataset shape %s" % Counter(y_train))
print("Resampled dataset shape %s" % Counter(y_res))

pca = PCA(n_components=4)
pca.fit(X_train)
X_pca = pca.transform(X_res)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_res, cmap="Paired", edgecolor="k", alpha=0.5)
plt.title("KNN-PCA with Random Undersampling")
plt.show()

clf = KNeighborsClassifier(n_neighbors=2, p=1)
clf.fit(X_res, y_res)

y_pred = clf.predict(X_test)

print("Accuracy Of UnderSampling %s" % accuracy_score(y_test, y_pred))
print("F1-score Of UnderSampling %s" % f1_score(y_test, y_pred, average=None))
draw_confusion_matrix(knn, X_test, y_pred)
print(classification_report(y_test, y_pred))
