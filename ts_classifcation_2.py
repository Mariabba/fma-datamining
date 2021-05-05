"""libraries"""
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    plot_confusion_matrix,
)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from music import MusicDB
import scipy.stats as stats

from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

"""
FILE 2  -CLASSIFICAZIONE A 8 CLASSI-GENERE BASE

In questo file vi sono 2 tipologie di classifcazione base per le TS:

2- Distance Based Class-KNN
3- Random Forest
"""

musi = MusicDB()
print(musi.df.info())

X = musi.df
y = musi.feat["enc_genre"]


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
        disp = plot_confusion_matrix(Clf, X, y, cmap="PuBuGn", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


"""KNN"""
clf = KNeighborsClassifier(n_neighbors=29, weights="distance", p=2)
clf.fit(X_train, y_train)
# Best parameters: {'n_neighbors': 29, 'p': 2, 'weights': 'distance'}

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
lb = LabelBinarizer()
lb.fit(y_test)
lb.classes_.tolist()

fpr = dict()
tpr = dict()
roc_auc = dict()
by_test = lb.transform(y_test)
by_pred = lb.transform(y_pred)
for i in range(8):
    fpr[i], tpr[i], _ = roc_curve(by_test[:, i], by_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    roc_auc = roc_auc_score(by_test, by_pred, average=None)

plt.figure(figsize=(8, 5))
for i in range(8):
    plt.plot(
        fpr[i],
        tpr[i],
        label="%s ROC curve (area = %0.2f)" % (lb.classes_.tolist()[i], roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title("Time Series Classifier KNN Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()


"""
Grid Search KNN

print("STA FACENDO LA GRIDSEARCH")
param_list = {
    "n_neighbors": list(np.arange(1, 30)),
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
# grid search
clf = KNeighborsClassifier()
grid_search = GridSearchCV(clf, param_grid=param_list)
grid_search.fit(X_train, y_train)

# results of the grid search
print("\033[1m" "Results of the grid search" "\033[0m")
print()
print("Best parameters: %s" % grid_search.best_params_)
print("Best estimator: %s" % grid_search.best_estimator_)
print()
print("Best k ('n_neighbors'): %s" % grid_search.best_params_["n_neighbors"])
print()

"""
