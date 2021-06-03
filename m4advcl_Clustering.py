import numpy as np
from matplotlib import pyplot as plt
from nltk.sem.logic import printtype
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
numeric_columns = [
    ("album", "comments"),
    ("album", "date_created"),
    ("album", "favorites"),
    ("album", "listens"),
    ("album", "type"),
    ("artist", "comments"),
    ("artist", "date_created"),
    ("artist", "favorites"),
    ("track", "comments"),
    ("track", "date_created"),
    ("track", "duration"),
    ("track", "favorites"),
    ("track", "interest"),
    ("track", "listens"),
    ("artist", "active_year_end"),
    ("artist", "wikipedia_page"),
    ("track", "composer"),
    ("track", "information"),
    ("track", "lyricist"),
    ("track", "publisher"),
    ("album", "engineer"),
    ("album", "information"),
    ("artist", "bio"),
    ("album", "producer"),
    ("artist", "website"),
]

X = df[numeric_columns].values
print("dataset:", X.shape)


"""MAKING ADVANCED CLUSTER X-MEANS
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
# score =
# print("Scores: '%s'" % str(score))


sil = silhouette(X, clusters).process().get_score()
print(len(sil))
print(sil.count(sil))
# Visual Guidotti
# print("score", score)
i = df.columns.values.tolist().index(("album", "listens"))
j = df.columns.values.tolist().index(("track", "favorites"))


sns.set()
for indexes in clusters:
    plt.scatter(X[indexes, i], X[indexes, j], alpha=0.4)
for c in centers:
    plt.scatter(c[i], c[j], s=100, edgecolors="k")

plt.show()


# ORIGINAL PCA

print(X.shape)
pca = PCA(n_components=25)
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


print("MI SONO ROTTA FACCIO L'OPTICS")
i = df.columns.values.tolist().index(("album", "listens"))
j = df.columns.values.tolist().index(("track", "favorites"))
optics = OPTICS(min_samples=5, max_eps=np.inf)
optics.fit(X)
print(optics.labels_[:10])

for cluster_id in np.unique(optics.labels_)[:20]:
    indexes = np.where(optics.labels_ == cluster_id)
    plt.scatter(X[indexes, i], X[indexes, j], alpha=0.4)


from sklearn.metrics import silhouette_score

silhouette5 = silhouette_score(X, optics.labels_)
print(silhouette5)
