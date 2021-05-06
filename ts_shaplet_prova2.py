"""libraries"""

import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    plot_confusion_matrix,
)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict, ShapeletModel

from music import MusicDB
import scipy.stats as stats

from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# librerie x shaplet


from pyts.classification import LearningShapelets
from pyts.utils import windowed_view

"""
FILE 1  -CLASSIFICAZIONE A 8 CLASSI-GENERE CON GLI SHAPELET

In questo file vi è la creazione degli shpalet con 2 tipologie di classifcazione:

1- shaplet base 
2- shaplet Distance Based Class RANDOM FOREST

"""


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Shaplet Classification Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Greens", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


# Carico il dataframe
musi = MusicDB()
print(musi.df.info())

X = musi.df
y = musi.feat["enc_genre"]  # classe targed ovvero genere con l'encoding

scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(X).reshape(X.shape[0], X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

# USO IL METODO

n_ts, ts_sz = X_train.shape
n_classes = len(set(y))

# Set the number of shapelets per size as done in the original paper
shapelet_sizes = {400: 24}

print("n_ts", n_ts)
print("ts_sz", ts_sz)
print("n_classes", n_classes)
print("shapelet_sizes", shapelet_sizes)

# Define the model using parameters provided by the authors (except that we use
# fewer iterations here)
st = ShapeletModel(
    n_shapelets_per_size=shapelet_sizes,
    optimizer="sgd",
    weight_regularizer=0.01,
    max_iter=1,
    verbose=1,
)


st.fit(X_train, y_train)
print("N_shaplet", len(st.shapelets_))
print("Eccoli", st.shapelets_)

# Apply on the training set
print("Apply  on the training set: \n")
Y_pred = st.predict(X_train)
print("Accuracy  %s" % accuracy_score(y_train, Y_pred))
print("F1-score %s" % f1_score(y_train, Y_pred, average=None))
print(classification_report(y_train, Y_pred))

# Apply on the test set and evaluate the performance
print("Apply on the test set and evaluate the performance: \n")
y_pred = st.predict(X_test)
print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score  %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

draw_confusion_matrix(st, X_test, y_test)

for s in st.shapelets_:
    plt.plot(s)

plt.show()


X_train2 = st.transform(X_train)
print("train shape:", X_train2.shape)
X_test2 = st.transform(X_test)
# Best: {'random_state': 2, 'min_samples_split': 3, 'min_samples_leaf': 20,
# 'max_features': 'log2', 'max_depth': 5, 'criterion': 'gini', 'class_weight': None}
clf_rf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=5,
    min_samples_split=3,
    min_samples_leaf=20,
    max_features="log2",
    class_weight=None,
    random_state=2,
)

clf_rf.fit(X_train2, y_train)

# Apply on the training set
print("Apply  on the training set-KNN: \n")
Y_pred2 = clf_rf.predict(X_train2)
print("Accuracy  %s" % accuracy_score(y_train, Y_pred2))
print("F1-score %s" % f1_score(y_train, Y_pred2, average=None))
print(classification_report(y_train, Y_pred2))

# Apply on the test set and evaluate the performance
print("Apply on the test set and evaluate the performance-KNN: \n")
y_pred2 = clf_rf.predict(X_test2)
print("Accuracy %s" % accuracy_score(y_test, y_pred2))
print("F1-score  %s" % f1_score(y_test, y_pred2, average=None))
print(classification_report(y_test, y_pred2))

draw_confusion_matrix(clf_rf, X_test2, y_test)
