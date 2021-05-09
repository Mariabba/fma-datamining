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
FILE 3  CLASSIFICAZIONE BINARIA GENERE CON GLI SHAPELET (SCEGLIERE LA CLASSE TARGET)

In questo file vi è la creazione degli shpalet con 3 tipologie di classifcazione:

1- shaplet base 
2- shaplet Distance Based Class-KNN
3- shaplet with Random Forest
"""

# Carico il dataframe
musi = MusicDB()
print(musi.df.info())

print(musi.feat["enc_genre"].unique())


X = musi.df
y = musi.feat["genre"]  # classe targed ovvero genere con l'encoding


scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(X).reshape(X.shape[0], X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Time Series Classifier KNN Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Reds", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


print(("\033[1m" "Making Undersampling" "\033[0m"))


# Classification
clf = KNeighborsClassifier(n_neighbors=5, p=1)
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

draw_confusion_matrix(clf, X_test, y_test)

"""ROC CURVE"""
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)

roc_auc = roc_auc_score(y_test, y_pred, average=None)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % (roc_auc))

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title("Linear SVM Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()
