import pandas as pd
from rich import pretty, print
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

import utils

pretty.install()

column2drop = [
    ("album", "title"),
    ("artist", "name"),
    ("track", "title"),
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "language_code"),
    ("track", "license"),
    ("track", "number"),
    ("track", "tags"),
    ("track", "genres"),  # todo da trattare se si vuole inserire solo lei
    ("track", "genres_all"),
]

all_dfs = utils.load_tracks_xyz(buckets="continuous", extractclass=("track", "listens"))

for df in all_dfs:
    try:
        all_dfs[df] = all_dfs[df].drop(column2drop, axis=1)
    except ValueError:
        pass

clf = GaussianNB()
clf.fit(all_dfs["train_x"], all_dfs["train_y"])
print(clf)
"""
y_pred = clf.predict(all_dfs["test_x"])

print("Accuracy %s" % accuracy_score(all_dfs["test_y"], y_pred))
print("F1-score %s" % f1_score(all_dfs["test_y"], y_pred, average=None))
print(classification_report(all_dfs["test_y"], y_pred))
"""
