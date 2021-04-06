import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import outlier as outlier
import pandas as pd
from pyod.models.abod import ABOD
from pyod.utils.data import get_outliers_inliers
from scipy.stats import stats

from sklearn.model_selection import GridSearchCV, train_test_split

import utils

df = utils.load("data/tracks.csv", dummies=True, buckets="basic", fill=True)


column2drop = [
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
    ("album", "date_created"),
]
df.drop(column2drop, axis=1, inplace=True)

# Riformatto le date

df["artist", "date_created"] = df["artist", "date_created"].astype("Int64")
# stampo qualche valore

print(df["artist", "date_created"].unique())
print(df["album", "listens"].unique())
print(df["album", "information"].unique())
column2drop = [
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
    ("artist", "date_created"),
    ("track", "date_recorded"),
    ("track", "date_created"),
]
df.drop(column2drop, axis=1, inplace=True)
df.info()

class_name = ("album", "type")
attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

# TRYING ABOD
print("Making Abod")
clf = ABOD(n_neighbors=10, contamination=0.011)

print("fitting")
clf.fit(X)

clf.decision_scores_

print("predicting")
outliers = clf.predict(X)

print("outliers:", len(outliers))
print(np.unique(outliers, return_counts=True))

plt.hist(clf.decision_scores_)
plt.axvline(np.min(clf.decision_scores_[np.where(outliers == 1)]), c="k")
plt.title("ABOD outliers identification")
plt.ylabel("Record")
plt.xticks(rotation=30)
plt.xlabel("")
plt.show()

# TRying to print better abod
x_outliers, x_inliers = get_outliers_inliers(X_train, y_train)

n_inliers = len(x_inliers)
n_outliers = len(x_outliers)

F1 = X_train[:, [0]].reshape(-1, 1)
F2 = X_train[:, [1]].reshape(-1, 1)

xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
# scatterplot
plt.scatter(F1, F2)

plt.title("Outliers Visualization")
plt.show()


print(outliers)
print(type(outliers))

miao = pd.Series(outliers)
print(miao)
miao.to_csv("strange_results/abod.csv")
