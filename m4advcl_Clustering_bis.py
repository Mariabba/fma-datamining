import numpy as np
from matplotlib import pyplot as plt
import utils
from pyclustering.cluster import xmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

df = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
)

df = df.head(100)
print(df.info())

"""
print(df.shape)
column2drop = [
    ("track", "language_code"),
    ("album", "type"),
]


df.drop(column2drop, axis=1, inplace=True)
"""

# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "type"),

]

for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le




numeric_columns = [
    ("album", "comments"),
    #("album", "date_created"),
    ("album", "favorites"),
    ("album", "listens"),
    ("artist", "comments"),
    ("album", "type"),
    #("artist", "date_created"),
    ("artist", "favorites"),
    ("track", "comments"),
    #("track", "date_created"),
    ("track", "duration"),
    ("track", "favorites"),
    ("track", "interest"),
    ("track", "listens"),
    #("artist", "active_year_end"),
    #("artist", "wikipedia_page"),
    #("track", "composer"),
    #("track", "information"),
    #("track", "lyricist"),
    #("track", "publisher"),
    #("album", "engineer"),
    #("album", "information"),
    #("artist", "bio"),
    #("album", "producer"),
    #("artist", "website"),
]

X = df[numeric_columns].values
print("dataset:", X.shape)
print(X)

xm = xmeans.xmeans(X)
xm.process()

clusters = xm.get_clusters()

centers = xm.get_centers()

print("Clusters: ", clusters)
print("Centers: ", centers)

# Visual Guidotti
i = df.columns.values.tolist().index(("album", "listens"))
j = df.columns.values.tolist().index(("track", "favorites"))


sns.set()
for indexes in clusters:
    plt.scatter(X[indexes, i], X[indexes, j], alpha=0.4)
for c in centers:
    plt.scatter(c[i], c[j], s=100, edgecolors="k")

plt.show()
