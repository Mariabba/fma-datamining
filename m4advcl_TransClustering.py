import numpy as np
from kmodes.kmodes import KModes

import utils

df = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="discrete", fill=True, outliers=True
)

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
df = df[numeric_columns]
print(df.info())
print(df.shape)


km = KModes(n_clusters=3, init="Huang", n_init=5, verbose=1)

clusters = km.fit_predict(df)


print("centroidi", km.cluster_centroids_)
lab = np.unique(km.labels_, return_counts=True)
print("le lables", lab)

# non funziona
# silhouette = silhouette_score(df, km.labels_)
# sil = silhouette_score(df, km.labels_, metric=pairwise_distances(df, km.labels_))
# print("sil:", sil)

sse = km.cost_
print("sse:", sse)
