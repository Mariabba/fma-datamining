import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from collections import defaultdict

from attr import attributes
from langdetect import detect
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from statsmodels.compat import pandas

import utils
from pathlib import Path

df = utils.load("../data/tracks.csv", dummies=True, buckets="basic", fill=True)

df.info()
print(Path.cwd())

"""
mi son apena resa conto che posso fare questa cosa solo con le variabili intere
quindi credo che farò cosi, di quelle selezionate le trasformo in intere e 
guardo i boxplot un attimo

# stampo qualche boxplot per avere più o meno un'idea del da farsi
fig = plt.subplots(figsize=(10, 10))
fig_dims = (1, 1)

ax = plt.subplot2grid(fig_dims, (0, 0))
df[("album", "comments")].plot.box(ax=ax)
plt.xlabel(("album", "comments"))
plt.show()
"""

column2drop = [
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "tags"),
    ("track", "genres"),
    ("track", "genres_all"),
    ("track", "license"),
    ("track", "language_code"),
]
df.drop(column2drop, axis=1, inplace=True)


# decido di eliminare  album title, artist name,set split e track title, ovviamente sono valori unici e outliers
# DEVO ELIMINARE GLI ID SENNO' COMBINANO MACELLO
column2drop = {
    ("album", "title"),
    ("artist", "name"),
    ("set", "split"),
    ("track", "title"),
    ("album", "id"),
    ("artist", "id"),
    ("album", "date_created"),
}
df.drop(column2drop, axis=1, inplace=True)

# Riformatto le date

df["artist", "date_created"] = df["artist", "date_created"].astype("Int64")

df.info()
print(df["artist", "date_created"].unique())
print(df["album", "listens"].unique())
print(df["album", "information"].unique())

class_name = ("album", "type")
attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

# NORMALIZZO I VALORI
from sklearn import preprocessing

x = df.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# FACCIO  IL PLOTTING BOXPLOT
plt.figure(figsize=(20, 25))
b = sns.boxplot(data=df, orient="v")
b.set(ylabel="Class", xlabel="Normalization Value")
plt.show()

# PLOT per trovare best esp

"""
sns.set()
neigh = NearestNeighbors(n_neighbors=3)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()
"""
"""APPLICO IL DBSCAN """
"""
from sklearn.cluster import DBSCAN

print("DBSCAN")
dbscan = DBSCAN(eps=4, min_samples=3)
dbscan = dbscan.fit(X)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

print(df.loc[(labels == -1)])

#Calcolo eps e min samples migliori
eps_to_test = [3, 4, 5, 6, 7, 8]  # i migliori sono esp 4 min 3
min_samples_to_test = [2, 3, 4, 5]

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
