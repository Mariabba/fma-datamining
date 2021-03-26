import numpy as np
import pandas as pd
from rich import pretty, print
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import utils

df = utils.load("data/tracks.csv", fill=True, buckets="continuous", dummies=True)

# debugging
del df[("album", "date_created")]
del df[("album", "tags")]
del df[("album", "title")]

del df[("artist", "tags")]
del df[("artist", "name")]

del df[("track", "genres")]
del df[("track", "genres_all")]
del df[("track", "language_code")]
del df[("track", "license")]
del df[("track", "tags")]
del df[("track", "title")]

del df[("set", "split")]

print(df.info())

class_name = ("album", "type")
attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

clf = IsolationForest(random_state=0)
clf.fit(X_train)

outliers = clf.predict(X_test)

print(np.unique(outliers, return_counts=True))
