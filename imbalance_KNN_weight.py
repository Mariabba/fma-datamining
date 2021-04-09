import itertools
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler
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
    LabelBinarizer,
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
from imblearn.over_sampling import SMOTE
from pandas import DataFrame

from pandas import DataFrame
import utils
from pathlib import Path


# FUNCTION
def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("KNN with ClassWeight confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="PuBu", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def conf_mat_disp(confusion_matrix, disp_labels):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=disp_labels
    )

    disp.plot(cmap="PuBu")


# DATASET
df = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
)

column2drop = [
    ("track", "language_code"),
]

df.drop(column2drop, axis=1, inplace=True)
print(df["album", "type"].unique())

# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "listens"),
    ("album", "type"),
    ("track", "license"),
    ("album", "comments"),
    ("album", "date_created"),
    ("album", "favorites"),
    ("artist", "comments"),
    ("artist", "date_created"),
    ("artist", "favorites"),
    ("track", "comments"),
    ("track", "date_created"),
    ("track", "duration"),
    ("track", "favorites"),
    ("track", "interest"),
    ("track", "listens"),
]
for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(df.info())

# Create KNN Object.
knn = KNeighborsClassifier(
    n_neighbors=5, p=1
)  # valori migliori dalla gridsearch n = 5, p=1, levarli per avere la standard

x = df.drop(columns=[("album", "type")])
y = df[("album", "type")]

X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)
print(X_train.shape, X_test.shape)

knn.fit(X_train, y_train)

# Apply the KNN on the test set and evaluate the performance
print("Apply the KNN on the test set and evaluate the performance: \n")
Y_pred = knn.predict(X_test)
print("Accuracy %s" % accuracy_score(y_test, Y_pred))
print("F1-score %s" % f1_score(y_test, Y_pred, average=None))
print(classification_report(y_test, Y_pred))


"""EMBALANCE LEARNING"""

"""CLASS WEIGHT"""
print("\033[1m" "Making KNN with Class Weight" "\033[0m")

clf = KNeighborsClassifier(n_neighbors=5, p=1, weights="distance")
clf.fit(X_train, y_train)

# Apply the knn on the training set
print("Apply the KNN-WEIGHT on the training set: \n")
y_pred = clf.predict(X_train)
print("Accuracy KNN-WEIGHT %s" % accuracy_score(y_train, y_pred))
print("F1-score KNN-WEIGHT %s" % f1_score(y_train, y_pred, average=None))
print(classification_report(y_train, y_pred))

# Apply the KNN on the test set and evaluate the performance
print("Apply the KNN-WEIGHT on the test set and evaluate the performance: \n")
y_pred = clf.predict(X_test)
print("Accuracy KNN-WEIGHT %s" % accuracy_score(y_test, y_pred))
print("F1-score KNN-WEIGHT %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))


draw_confusion_matrix(clf, X_test, y_test)

"""ROC Curve"""

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
plt.title("KNN-Weight Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()
