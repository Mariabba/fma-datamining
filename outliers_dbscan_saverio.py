from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langdetect import detect
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from statsmodels.compat import pandas

import utils

df = utils.load_tracks(buckets="continuous", outliers=False)

column2drop = [
    ("track", "language_code"),
    ("track", "license"),
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

df.drop(column2drop, axis=1, inplace=True)

print(df.info())

"""
# FACCIO  IL PLOTTING BOXPLOT del Df completo
plt.figure(figsize=(20, 25))
b = sns.boxplot(data=df, orient="h")
b.set(ylabel="Class", xlabel="Normalization Value")
plt.show()


# PLOT per trovare best esp
sns.set()
neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()
"""
"""APPLICO IL DBSCAN """

X = df.drop(columns=[("album", "type")])
y = df[("album", "type")]

print("DBSCAN")
dbscan = DBSCAN(eps=5000, min_samples=24)

dbscan = dbscan.fit(X)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

print(df.loc[(labels == -1)])
df["cluster"] = labels
df["cluster"].to_csv("strange_results_new/4000dbscan.csv")

"""
# Calcolo eps e min samples migliori
eps_to_test = [100, 200, 300, 400, 500, 600]  # i migliori sono esp 4 min 3
min_samples_to_test = [24]

print("EPS:", list(eps_to_test))
print("MIN_SAMPLES:", list(min_samples_to_test))


def get_metrics(eps, min_samples, dataset, iter_):
    # Fitting ======================================================================

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(dataset)

    # Mean Noise Point Distance metric =============================================
    noise_indices = dbscan.labels_ == -1

    if True in noise_indices:
        neighboors = NearestNeighbors(n_neighbors=6).fit(dataset)
        distances, indices = neighboors.kneighbors(dataset)
        noise_distances = distances[noise_indices, 1:]
        noise_mean_distance = round(noise_distances.mean(), 3)
    else:
        noise_mean_distance = None

    # Number of found Clusters metric ==============================================

    number_of_clusters = len(set(dbscan.labels_[dbscan.labels_ >= 0]))

    # Log ==========================================================================

    print(
        "%3d | Tested with eps = %3s and min_samples = %3s | %5s %4s"
        % (iter_, eps, min_samples, str(noise_mean_distance), number_of_clusters)
    )

    return (noise_mean_distance, number_of_clusters)


# Dataframe per la metrica sulla distanza media dei noise points dai K punti più vicini
results_noise = pd.DataFrame(
    data=np.zeros((len(eps_to_test), len(min_samples_to_test))),  # Empty dataframe
    columns=min_samples_to_test,
    index=eps_to_test,
)

# Dataframe per la metrica sul numero di cluster
results_clusters = pd.DataFrame(
    data=np.zeros((len(eps_to_test), len(min_samples_to_test))),  # Empty dataframe
    columns=min_samples_to_test,
    index=eps_to_test,
)

iter_ = 0

print("ITER| INFO%s |  DIST    CLUS" % (" " * 39))
print("-" * 65)

for eps in eps_to_test:
    for min_samples in min_samples_to_test:
        iter_ += 1

        # Calcolo le metriche
        noise_metric, cluster_metric = get_metrics(eps, min_samples, df, iter_)

        # Inserisco i risultati nei relativi dataframe
        results_noise.loc[eps, min_samples] = noise_metric
        results_clusters.loc[eps, min_samples] = cluster_metric

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sns.heatmap(results_noise, annot=True, ax=ax1, cbar=False).set_title(
    "METRIC: Mean Noise Points Distance"
)
sns.heatmap(results_clusters, annot=True, ax=ax2, cbar=False).set_title(
    "METRIC: Number of clusters"
)

ax1.set_xlabel("N")
ax2.set_xlabel("N")
ax1.set_ylabel("EPSILON")
ax2.set_ylabel("EPSILON")

plt.tight_layout()
plt.show()
"""
"""
#funzione per plottare il df in 2 dimensioni
pca = PCA(n_components=2)
pca.fit(X)
X_train_pca = pca.transform(X)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
plt.show()
"""
