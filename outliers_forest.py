import numpy as np
from rich import print
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import utils

df = utils.load_tracks(buckets="continuous")

del df[("track", "language_code")]
del df[("track", "license")]

class_name = ("album", "type")
attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

clf = IsolationForest(random_state=3549)
clf.fit(X_train)
outliers = clf.predict(X_test)

counts = np.unique(outliers, return_counts=True)
percentage = counts[1][0] / (counts[1][1] + counts[1][0]) * 100
print(
    f"Of the {counts[1][1] + counts[1][0]} test records, I found {counts[1][0]} outliers: that is the {percentage:.2f} %."
)
