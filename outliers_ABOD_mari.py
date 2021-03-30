from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import utils
from pathlib import Path
from pyod.models.abod import ABOD
import matplotlib.pyplot as plt

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
clf = ABOD(method="default")
clf.fit(X)
clf.decision_scores_
outliers = clf.predict(X)
print("outliers:", len(outliers))
print(np.unique(outliers, return_counts=True))

plt.hist(clf.decision_scores_, bins=20)
plt.axvline(np.min(clf.decision_scores_[np.where(outliers == 1)]), c="k")
plt.show()
