"""libraries"""

import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
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
from tslearn.utils import ts_size

from music import MusicDB
import scipy.stats as stats
from random import sample
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# librerie x shaplet
from keras.optimizers import Adagrad
from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict

from pyts.classification import LearningShapelets
from pyts.utils import windowed_view

# Carico il dataframe
musi = MusicDB()
print(musi.df.info())

X = musi.df
y = musi.feat["enc_genre"]  # classe targed ovvero genere con l'encoding

"""Creazione shaplet"""
# versione 1
n_ts, ts_sz = X.shape
n_classes = len(set(y))

# Set the number of shapelets per size as done in the original paper
shapelet_sizes = grabocka_params_to_shapelet_size_dict(
    n_ts=n_ts, ts_sz=ts_sz, n_classes=n_classes, l=0.1, r=1
)

print("n_ts", n_ts)
print("ts_sz", ts_sz)
print("n_classes", n_classes)
print("shapelet_sizes", shapelet_sizes)

# Define the model using parameters provided by the authors (except that we use
# fewer iterations here)
shp_clf = ShapeletModel(
    n_shapelets_per_size=shapelet_sizes,
    optimizer="sgd",
    weight_regularizer=0.01,
    max_iter=50,
    verbose=1,
)

shp_clf.fit(X, y)

predicted_labels = shp_clf.predict(X)
print("Correct classification rate:", accuracy_score(y, predicted_labels))

predicted_locations = shp_clf.locate(X)

"""plot shaplet e dataset"""
ts_id = 0
plt.figure()
n_shapelets = sum(shapelet_sizes.values())
plt.title(
    "Example locations of shapelet matches "
    "(%d shapelets extracted)".format(n_shapelets)
)

plt.plot(X[ts_id].ravel(), label="Time Series Dataset")
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[ts_id, idx_shp]
    plt.plot(np.arange(t0, t0 + len(shp)), shp, linewidth=2)
plt.legend()
plt.show()

"""plot tslearn solo shaplet"""

# Make predictions and calculate accuracy score

# Plot the different discovered shapelets
plt.figure()
for i, sz in enumerate(shapelet_sizes.keys()):
    plt.subplot(len(shapelet_sizes), 1, i + 1)
    plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
    for shp in shp_clf.shapelets_:
        if ts_size(shp) == sz:
            plt.plot(shp.ravel(), label=" Shaplet ")
    plt.xlim([0, max(shapelet_sizes.keys()) - 1])

plt.tight_layout()
plt.legend()
plt.show()

"""plot singoli shaplet"""

sel_shapelets = np.asarray(
    [
        shp_clf.shapelets_[0],
        shp_clf.shapelets_[1],
        shp_clf.shapelets_[2],
        shp_clf.shapelets_[3],
        shp_clf.shapelets_[4],
        shp_clf.shapelets_[5],
        shp_clf.shapelets_[6],
        shp_clf.shapelets_[7],
    ],
)

fig, axs = plt.subplots(4, 2, figsize=(10, 12))

axs[0, 0].plot(sel_shapelets[0], color="blue")
axs[0, 0].set_title("shaplet 0")

axs[0, 1].plot(sel_shapelets[1], color="orange")
axs[0, 1].set_title("shaplet 1")

axs[1, 0].plot(sel_shapelets[2], color="green")
axs[1, 0].set_title("shaplet 2")

axs[1, 1].plot(sel_shapelets[3], color="red")
axs[1, 1].set_title("shaplet 3")

axs[2, 0].plot(sel_shapelets[4], color="purple")
axs[2, 0].set_title("shaplet 4")

axs[2, 1].plot(sel_shapelets[5], color="brown")
axs[2, 1].set_title("shaplet 5")

axs[3, 0].plot(sel_shapelets[6], color="pink")
axs[3, 0].set_title("shaplet 6")

axs[3, 1].plot(sel_shapelets[7], color="gray")
axs[3, 1].set_title("shaplet 7")

fig.tight_layout()
plt.show()
