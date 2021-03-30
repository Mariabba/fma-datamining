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

df = utils.load("data/tracks.csv", dummies=True, buckets="basic", fill=True)

df.info()

column2drop = [
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "tags"),
    ("track", "genres"),
    ("track", "genres_all"),
    ("track", "license"),
    ("track", "language_code"),
    ("album", "title"),
    ("artist", "name"),
    ("set", "split"),
    ("track", "title"),
    ("album", "id"),
    ("artist", "id"),
    ("album", "date_created"),
]
df.drop(column2drop, axis=1, inplace=True)

# Riformatto le date

df["artist", "date_created"] = df["artist", "date_created"].astype("Int64")
# stampo qualche valore
df.info()
print(df["artist", "date_created"].unique())
print(df["album", "listens"].unique())
print(df["album", "information"].unique())


"""BOXPLOT PER OGNUNA DELLE 26 VARIABILI CHE VOGLIO TENERE 
df[("album", "comments")].plot.box()
plt.xlabel(("album", "comments"))
plt.show()
df[("album", "favorites")].plot.box()
plt.xlabel(("album", "favorites"))
plt.show()
df[("album", "listens")].plot.box()
plt.xlabel(("album", "listens"))
plt.show()
df[("album", "tracks")].plot.box()
plt.xlabel(("album", "tracks"))
plt.show()
df[("album", "engineer")].plot.box() #da levare
plt.xlabel(("album", "engineer"))
plt.show()
df[("album", "information")].plot.box() #da levare
plt.xlabel(("album", "information"))
plt.show()
df[("album", "producer")].plot.box() #da levare
plt.xlabel(("album", "producer"))
plt.show()
df[("album", "type")].plot.box()
plt.xlabel(("album", "type"))
plt.show()

df[("artist", "comments")].plot.box()
plt.xlabel(("artist", "comments"))
plt.show()
df[("artist", "date_created")].plot.box()
plt.xlabel(("artist", "date_created"))
plt.show()
df[("artist", "favorites")].plot.box()
plt.xlabel(("artist", "favorites"))
plt.show()
df[("artist", "active_year_end")].plot.box() #da levare
plt.xlabel(("artist", "active_year_end"))
plt.show()
df[("artist", "wikipedia_page")].plot.box() #da levare
plt.xlabel(("artist", "wikipedia_page"))
plt.show()
df[("artist", "bio")].plot.box() #da levare
plt.xlabel(("artist", "bio"))
plt.show()
df[("artist", "website")].plot.box() #dalevare
plt.xlabel(("artist", "website"))
plt.show()

df[("track", "comments")].plot.box()
plt.xlabel(("track", "comments"))
plt.show()
df[("track", "date_created")].plot.box()
plt.xlabel(("track", "date_created"))
plt.show()
df[("track", "duration")].plot.box()
plt.xlabel(("track", "duration"))
plt.show()
df[("track", "favorites")].plot.box()
plt.xlabel(("track", "favorites"))
plt.show()
df[("track", "interest")].plot.box()
plt.xlabel(("track", "interest"))
plt.show()
df[("track", "listens")].plot.box()
plt.xlabel(("track", "listens"))
plt.show()
df[("track", "number")].plot.box()
plt.xlabel(("track", "number"))
plt.show()
df[("track", "composer")].plot.box() #dalevare
plt.xlabel(("track", "composer"))
plt.show()
df[("track", "date_recorded")].plot.box()#dalevare
plt.xlabel(("track", "date_recorded"))
plt.show()
df[("track", "information")].plot.box()#da levare
plt.xlabel(("track", "information"))
plt.show()
df[("track", "lyricist")].plot.box() #da levare
plt.xlabel(("track", "lyricist"))
plt.show()
df[("track", "publisher")].plot.box() #da levare
plt.xlabel(("track", "publisher"))
plt.show()
"""

# elimino le colonne che ho selezionato
column2drop = [
    ("album", "engineer"),
    ("album", "information"),
    ("album", "producer"),
    ("artist", "active_year_end"),
    ("artist", "wikipedia_page"),
    ("artist", "website"),
    ("artist", "bio"),
    ("track", "composer"),
    ("track", "date_recorded"),
    ("track", "information"),
    ("track", "lyricist"),
    ("track", "publisher"),
    ("artist", "date_created"),
    ("track", "date_recorded"),
    ("track", "date_created"),
    ("album", "type"),
]
df.drop(column2drop, axis=1, inplace=True)
# normalizzo in basic
"""
normalized_df = (df - df.min()) / (df.max() - df.min())
df = normalized_df
"""


def normalize(feature):
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])


for col in df.columns:
    normalize(col)

attributes = [col for col in df.columns]
X = df[attributes].values
print(df.info())
"""
# FACCIO  IL PLOTTING BOXPLOT del Df completo
plt.figure(figsize=(20, 25))
b = sns.boxplot(data=df, orient="h")
b.set(ylabel="Class", xlabel="Normalization Value")
plt.show()


# PLOT per trovare best esp
sns.set()
neigh = NearestNeighbors(n_neighbors=100)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:, 10]
plt.plot(distances)
plt.show()
"""
"""APPLICO IL DBSCAN """


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
miao = df.loc[(labels == -1)]
miao = miao["album", "comments"]
miao.to_csv("4000dbscan.csv")

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
