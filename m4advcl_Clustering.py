import seaborn as sns
from matplotlib import pyplot as plt
from pyclustering.cluster import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from utils import TracksMetaDB

df = TracksMetaDB(buckets="continuous").normalized


# df = utils.load_tracks(
#    "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
# )


# column2drop = [
#    ("track", "language_code"),
# ]
# df.drop(column2drop, axis=1, inplace=True)
"""
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
"""
X = df.values
print("dataset:", X.shape)

"""MAKING ADVANCED CLUSTER X-MEANS"""
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(X, amount_initial_centers).initialize()

xmeans_instance = xmeans.xmeans(X, initial_centers, kmax=5)
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
"""COMBINAZIONE DEI CLUSTER"""
print()

a = df.columns.values.tolist().index(("album", "comments"))
b = df.columns.values.tolist().index(("album", "date_created"))
cc = df.columns.values.tolist().index(("album", "favorites"))
d = df.columns.values.tolist().index(("album", "listens"))
e = df.columns.values.tolist().index(("artist", "comments"))
f = df.columns.values.tolist().index(("artist", "date_created"))
g = df.columns.values.tolist().index(("artist", "favorites"))
h = df.columns.values.tolist().index(("track", "date_created"))
i = df.columns.values.tolist().index(("track", "duration"))
j = df.columns.values.tolist().index(("track", "favorites"))
k = df.columns.values.tolist().index(("track", "interest"))
l = df.columns.values.tolist().index(("track", "listens"))

k = 4
l = 8
print(
    "a:",
    a,
    "b:",
    b,
    "cc:",
    cc,
    "d:",
    d,
    "e:",
    e,
    "f:",
    f,
    "g:",
    g,
    "h:",
    h,
    "i:",
    i,
    "j:",
    j,
    "k:",
    k,
    "l:",
    l,
)

sns.set()
for indexes in clusters:
    plt.scatter(X[indexes, h], X[indexes, l], alpha=0.4)
for c in centers:
    plt.scatter(c[h], c[l], s=100, edgecolors="k")
plt.xlabel("track,date_created")
plt.ylabel("track,listens")
plt.title("Visualizing centroids and centers")
plt.show()


sns.set()
for indexes in clusters:
    plt.scatter(X[indexes, g], X[indexes, cc], alpha=0.4)
for c in centers:
    plt.scatter(c[g], c[cc], s=100, edgecolors="k")
plt.ylabel("artist,favorites")
plt.xlabel("album,favourites")
plt.title("Visualizing centroids and centers")
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
