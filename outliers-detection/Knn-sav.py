import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# sklearn
from pyod.utils.example import visualize
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from pyod.models.knn import KNN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# DATASET
df = utils.load("../data/tracks.csv", dummies=True, buckets="continuous", fill=True)
column2drop = [
    ("album", "type"),
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "tags"),
    ("track", "genres"),
    ("track", "genres_all"),
    ("track", "license"),
    ("track", "language_code"),
    ("album", "title"),
    ("artist", "name"),
    ("set", "split"),
    ("track", "title"),
    ("album", "id"),
    ("artist", "id"),
    ("album", "engineer"),
    ("album", "information"),
    ("album", "producer"),
    ("artist", "active_year_end"),
    ("artist", "wikipedia_page"),
    ("artist", "website"),
    ("artist", "bio"),
    ("track", "composer"),
    ("track", "date_recorded"),
    ("track", "information"),
    ("track", "lyricist"),
    ("track", "publisher"),
    ("album", "date_created"),
    ("artist", "date_created"),
    ("track", "date_recorded"),
    ("track", "date_created"),
]
df.drop(column2drop, axis=1, inplace=True)
print(df.info())

def normalize(feature):
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])


for col in df.columns:
    normalize(col)

"""
# FACCIO  IL PLOTTING BOXPLOT del Df completo
plt.figure(figsize=(20, 25))
b = sns.boxplot(data=df, orient="h")
b.set(ylabel="Class", xlabel="Normalization Value")
plt.show()
"""

X = df.values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

# instantiate model
nbrs = NearestNeighbors(n_neighbors = 3)
# fit model
nbrs.fit(X)

# distances and indexes of k-neaighbors from model outputs
distances, indexes = nbrs.kneighbors(X)
# plot mean of k-distances of each observation
plt.plot(distances.mean(axis =1))
plt.show()

# visually determine cutoff values > 10
outlier_index = np.where(distances.mean(axis = 1) > 0.8)
print(outlier_index)

# filter outlier values
outlier_values = df.iloc[outlier_index]
print(outlier_values.describe())
print(outlier_values.value_counts())
print(outlier_values.info())


"""
clf = KNN()
clf.fit(X)


print(clf.decision_scores_)


anomlay_score = clf.decision_function(X)

outliers = clf.predict(X)
np.unique(outliers, return_counts=True)

np.mean(anomlay_score[np.where(outliers==1)])

np.mean(anomlay_score[np.where(outliers==0)])

np.mean(clf.decision_scores_[np.where(outliers==1)])

np.mean(clf.decision_scores_[np.where(outliers==0)])

plt.hist(clf.decision_scores_, bins=20)
plt.axvline(np.min(clf.decision_scores_[np.where(outliers==1)]), c='k')
plt.show()

#funzione per plottare il df in 2 dimensioni
pca = PCA(n_components=2)
pca.fit(X)
X_train_pca = pca.transform(X)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
plt.show()
"""