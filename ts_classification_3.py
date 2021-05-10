"""libraries"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.generators import random_walks
from tslearn.piecewise import SymbolicAggregateApproximation
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
from music import MusicDB
import scipy.stats as stats
from collections import Counter

from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

"""
FILE 3-  CLASSIFICAZIONE CON APPROSSIMAZIONE CON SAX
"""

# Carico il dataframe
musi = MusicDB()
print(musi.df.info())

print(musi.feat["enc_genre"].unique())


X_no = musi.df
y = musi.feat["enc_genre"]  # classe targed ovvero genere con l'encoding

# normalizzazione con mean variance
scaler = TimeSeriesScalerMeanVariance()
X_no = pd.DataFrame(
    scaler.fit_transform(musi.df.values).reshape(
        musi.df.values.shape[0], musi.df.values.shape[1]
    )
)
X_no.index = musi.df.index

# approssimazione con sax

sax = SymbolicAggregateApproximation(n_segments=155, alphabet_size_avg=10)
X1 = sax.fit_transform(X_no)
print(X1.shape)
X = np.squeeze(X1)
print(X.shape)

# Classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)
"""INSERISCO PARTE SHAPLET"""
n_ts, ts_sz = X_train.shape
n_classes = len(set(y))

# shapelet_sizes = grabocka_params_to_shapelet_size_dict(
#   n_ts=n_ts, ts_sz=ts_sz, n_classes=n_classes, l=0.1, r=1
# )
# 6 shaplet da 15 con grabocka

shapelet_sizes = {15: 24}
print("n_ts", n_ts)
print("ts_sz", ts_sz)
print("n_classes", n_classes)
print("shapelet_sizes", shapelet_sizes)

shp_clf = ShapeletModel(
    n_shapelets_per_size=shapelet_sizes,
    optimizer="sgd",
    weight_regularizer=0.01,
    max_iter=50,
    verbose=1,
)

shp_clf.fit(X_train, y_train)
print("Apply on the test set and evaluate the performance: \n")
y_pred = shp_clf.predict(X_test)
print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score  %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))


"""CLASSIFICATORE"""
print("KNN- Shaplet Based")
X_train2 = shp_clf.transform(X_train)
print("train shape:", X_train2.shape)
X_test2 = shp_clf.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=15, weights="uniform", p=1)
# Best parameters: {'n_neighbors': 19, 'p': 2, 'weights': 'uniform'}
knn.fit(X_train2, y_train)

# Apply on the training set
print("Apply  on the training set-KNN: \n")
Y_pred = knn.predict(X_train2)
print("Accuracy  %s" % accuracy_score(y_train, Y_pred))
print("F1-score %s" % f1_score(y_train, Y_pred, average=None))
print(classification_report(y_train, Y_pred))

# Apply on the test set and evaluate the performance
print("Apply on the test set and evaluate the performance-KNN: \n")
y_pred = knn.predict(X_test2)
print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score  %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

"""
print("STA FACENDO LA GRIDSEARCH")
param_list = {
    "n_neighbors": list(np.arange(1, 20)),
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
# grid search
clf = KNeighborsClassifier()
grid_search = GridSearchCV(clf, param_grid=param_list)
grid_search.fit(X_train2, y_train)

# results of the grid search
print("\033[1m" "Results of the grid search" "\033[0m")
print()
print("Best parameters: %s" % grid_search.best_params_)
print("Best estimator: %s" % grid_search.best_estimator_)
print()
print("Best k ('n_neighbors'): %s" % grid_search.best_params_["n_neighbors"])
print()
"""
