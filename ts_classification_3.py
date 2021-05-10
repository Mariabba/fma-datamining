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
from sklearn.model_selection import train_test_split, cross_val_score
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

# approssimazione con sax
n_sax_symbols = 50
sax = SymbolicAggregateApproximation(n_segments=20, alphabet_size_avg=n_sax_symbols)
ts_sax_df = sax.fit_transform(musi.df)
sax_dataset_inv = sax.inverse_transform(ts_sax_df)

X = ts_sax_df
y = musi.feat["enc_genre"]  # classe targed ovvero genere con l'encoding


scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(X).reshape(X.shape[0], X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

# Classification
clf = KNeighborsClassifier(n_neighbors=5, p=1, weights="distance")
clf.fit(X_train, y_train)


# Apply on the training set
print("Apply  on the training set: \n")
Y_pred = clf.predict(X_train)
print("Accuracy  %s" % accuracy_score(y_train, Y_pred))
print("F1-score %s" % f1_score(y_train, Y_pred, average=None))
print(classification_report(y_train, Y_pred))

# Apply on the test set and evaluate the performance
print("Apply on the test set and evaluate the performance: \n")
y_pred = clf.predict(X_test)
print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score  %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
