import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.abod import ABOD
from pyod.utils.data import get_outliers_inliers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import utils

df = utils.load_tracks(dummies=True, buckets="continuous", fill=True, outliers=False)
df.info()

column2drop = [
    ("track", "language_code"),
    ("track", "license"),
    ("artist", "wikipedia_page"),
    ("track", "composer"),
    ("track", "information"),
    ("track", "lyricist"),
    ("track", "publisher"),
    ("album", "engineer"),
    ("album", "information"),
    ("artist", "bio"),
    ("album", "producer"),
    ("artist", "website"),
]

df.drop(column2drop, axis=1, inplace=True)

# Riformatto le date

# encoding
label_encoders = dict()
column2encode = []
for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


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

# Trying to print better abod
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

miao = pd.Series(outliers, index=df.index)
print(miao)
miao.to_csv("strange_results_new/abod.csv")
