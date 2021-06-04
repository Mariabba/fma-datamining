import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from pyclustering.cluster import xmeans, cluster_visualizer, cluster_visualizer_multidim
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.silhouette import silhouette
from pyclustering.utils import read_sample
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from array import array
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from sklearn.cluster import OPTICS
import utils

df = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
)


print(df.shape)
column2drop = [
    ("track", "language_code"),
]
df.drop(column2drop, axis=1, inplace=True)

numeric_columns = [
    ("album", "comments"),
    ("album", "date_created"),
    ("album", "favorites"),
    ("album", "listens"),
    ("artist", "comments"),
    # ("album", "type"),
    ("artist", "date_created"),
    ("artist", "favorites"),
    # ("track", "comments"),
    ("track", "date_created"),
    ("track", "duration"),
    ("track", "favorites"),
    ("track", "interest"),
    ("track", "listens"),
]

X = df[numeric_columns].values
print("dataset:", X.shape)


"""MAKING ADVANCED CLUSTER X-MEANS"""
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(X, amount_initial_centers).initialize()

xmeans_instance = xmeans.xmeans(X, initial_centers)
xmeans_instance.process()

# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
clus2 = xmeans_instance.get_cluster_encoding()
# a_cluster = np.asarray(clusters)
# a_centers = np.asarray(centers, dtype=object)
# print("dopo cluster", type(a_cluster))
# print("dopo centers", type(a_centers))
print(clus2)
print("count cluster", clusters.count(clusters))
print("count centers", centers.count(centers))

# Calculate Silhouette score
# print("sil making")
# print("Scores: '%s'" % str(score))

print("SSE: ", xmeans.xmeans.get_total_wce(xmeans_instance))
# score = silhouette(X, clusters).process().get_score()
# print("Score SIl:", score)

# score = silhouette_score(xmeans_instance, clusters)
# print(score)

i = df.columns.values.tolist().index(("album", "listens"))
j = df.columns.values.tolist().index(("track", "favorites"))

sns.set()
colours = ListedColormap(["r", "b", "g"])
for indexes in clusters:
    plt.scatter(X[indexes, i], X[indexes, j], alpha=0.4, cmap=colours)
for c in centers:
    plt.scatter(c[i], c[j], s=100, edgecolors="k")
plt.xlabel("album,listens")
plt.ylabel("track,favourites")
plt.title("Visualizing centeroids and centers")
plt.show()

"""
# ORIGINAL PCA

print(X.shape)
pca = PCA(n_components=4)
pca.fit(X)
X_pca = pca.transform(X)
print("pcs shape", X_pca.shape)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    cmap="Set2",
    edgecolor="k",
    alpha=0.5,
)
plt.title("Clustering PCA")
plt.show()
"""
