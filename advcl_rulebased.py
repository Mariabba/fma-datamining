import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from collections import defaultdict
import utils
import wittgenstein as lw

from sklearn.model_selection import train_test_split
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


class_name = ('album', 'type')

df = utils.load_tracks(buckets="discrete")
df = df.head(1000)

column2drop = [
    # ("track", "license"),
    # ("track", "language_code"),
]
df.drop(column2drop, axis=1, inplace=True)

print(df.info())
#df['album', 'type'] = df['album', 'type'].replace(['Single Tracks', 'Live Performance', 'Radio Program'],
#                                                  ['NotAlbum', 'NotAlbum', 'NotAlbum'])

# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "type"),
]

for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
dfY = df[class_name]
df = pd.concat([dfX, dfY], axis=1)
print(df.info())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

ripper_clf = lw.RIPPER()

ripper_clf.fit(X_train, y_train, class_feat=("album", "type"), pos_class=0, k=1, prune_size=0.33)

print(ripper_clf)
print(ripper_clf.ruleset_)

# Collect performance metrics
precision = ripper_clf.score(X_test, y_test, precision_score)
recall = ripper_clf.score(X_test, y_test, recall_score)
cond_count = ripper_clf.ruleset_.count_conds()
print(ripper_clf.ruleset_.out_pretty())
print(f'precision: {precision} recall: {recall} conds: {cond_count}')

"""
# grid search
param_grid = {"prune_size": [0.33, 0.5], "k": [1, 2], "class_feat": ["album", "type"], "pos_class": [0]}
grid = GridSearchCV(estimator=ripper_clf, param_grid=param_grid)
grid.fit(X_train, y_train)
clf = grid.best_estimator_
print(report(grid.cv_results_, n_top=3))
"""