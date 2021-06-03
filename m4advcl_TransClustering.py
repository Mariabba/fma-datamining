from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
import utils
from sklearn.metrics import pairwise_distances

df = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="discrete", fill=True, outliers=True
)

print(df.shape)
column2drop = [
    ("track", "language_code"),
]

km = KModes(n_clusters=4, init="Huang", n_init=5, verbose=1)

clusters = km.fit_predict(df)


print("centroidi", km.cluster_centroids_)

print("le lables", km.labels_)

# non funziona
# silhouette = silhouette_score(df, km.labels_)
# sil = silhouette_score(df, km.labels_, metric=pairwise_distances(df, km.labels_))
# print("sil:", sil)

sse = km.cost_
print("sse:", sse)
