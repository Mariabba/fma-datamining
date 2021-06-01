import numpy as np
from matplotlib import pyplot as plt
from nltk.sem.logic import printtype
from pyclustering.cluster import xmeans, cluster_visualizer, cluster_visualizer_multidim
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.silhouette import silhouette
from sklearn.preprocessing import LabelEncoder
from array import array
import utils

df = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
)

print(df.shape)
column2drop = [
    ("track", "language_code"),
]

df.drop(column2drop, axis=1, inplace=True)
# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "listens"),
    ("album", "type"),
    ("track", "license"),
    ("album", "comments"),
    ("album", "date_created"),
    ("album", "favorites"),
    ("artist", "comments"),
    ("artist", "date_created"),
    ("artist", "favorites"),
    ("track", "comments"),
    ("track", "date_created"),
    ("track", "duration"),
    ("track", "favorites"),
    ("track", "interest"),
    ("track", "listens"),
]
for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


print(df.info())

"""MAKING ADVANCED CLUSTER"""
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(
    df.values, amount_initial_centers
).initialize()

xmeans_instance = xmeans.xmeans(df.values, initial_centers, 20)
xmeans_instance.process()


# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
res = array("f", clusters)
print(type(clusters))


plt.plot(clusters)
plt.show()


# Calculate Silhouette score
# score = silhouette(df.values, clusters).process().get_score()
# score.plot()


"""Visual Guidotti"""
# print("score", score)
i = df.columns.values.tolist().index(("album", "listens"))
j = df.columns.values.tolist().index(("track", "favorites"))

for indexes in clusters:
    plt.scatter(df.values[indexes, i], df.values[indexes, j], alpha=0.4)
for c in centers:
    plt.scatter(c[i], c[j], s=100, edgecolors="k")
plt.show()
