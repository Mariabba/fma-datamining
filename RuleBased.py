import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from collections import defaultdict
import utils
import wittgenstein as lw

from sklearn.model_selection import train_test_split

class_name = ("album", "type")

df = utils.load_tracks(buckets="discrete")
column2drop = [("track", "license")]
df.drop(column2drop, axis=1, inplace=True)

print(df.info())
df["album", "type"] = df["album", "type"].replace(
    ["Single Tracks", "Live Performance", "Radio Program"],
    ["NotAlbum", "NotAlbum", "NotAlbum"],
)

attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep="=")
dfY = df[class_name]
df = pd.concat([dfX, dfY], axis=1)
print(df.info())
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)


ripper_clf = lw.RIPPER()


ripper_clf.fit(X_train, y_train, feature_names=attributes)

print(ripper_clf)
print(ripper_clf.ruleset_)
print(ripper_clf.score(X_test, y_test))
"""
