import pandas as pd
from rich import pretty, print
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import numpy as np

import utils

df = utils.load("data/tracks.csv", fill=True, buckets="continuous", dummies=True)

# debugging


class_name = ("album", "d_type")
attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)
print(df.info())

print(y.unique())

"""
clf = IsolationForest(random_state=0)
clf.fit(X_train)

outliers = clf.predict(X_test)

np.unique(outliers, return_counts=True)
"""
