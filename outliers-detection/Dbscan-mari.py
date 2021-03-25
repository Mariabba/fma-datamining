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

df = utils.load("../data/tracks.csv", dummies=True)

df.info()

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
"""
#==============trasformo in interi gli object==========
Queste son le variabili ora ci ragiono un attimo su ognuna
 (album, tags)                99404 non-null  object  
 (artist, tags)               99404 non-null  object  
 (track, genres)              99404 non-null  object  
 (track, genres_all)          99404 non-null  object  
 (track, tags)                99404 non-null  object 
 
Sicuramente devo eliminare i tags ma prima provo a trasformarne uno e farci il boxplot
per vedere cosa ne esce,nulla di buono, non posso trasformarli in int quindi elimino tutto.
Per quanto riguarda genres è un problema quindi le elimino
"""
column2drop = [
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "tags"),
    ("track", "genres"),
    ("track", "genres_all"),
]
df.drop(column2drop, axis=1, inplace=True)

"""
#==============trasformo in interi le category========================
(album, information)         81260 non-null  category
(album, type)                99404 non-null  category
(artist, bio)                66610 non-null  category 
(track, license)             99404 non-null  category
"""
df["album", "information"] = (~df["album", "information"].isnull()).astype(int)
df["album", "type"] = (~df["album", "type"].isnull()).astype(int)
df["artist", "bio"] = (~df["artist", "bio"].isnull()).astype(int)
df["track", "license"] = (~df["track", "license"].isnull()).astype(int)

"""
============trasformo in interi le stringhe=======================
(album, engineer)
(album, producer)            17707 non-null  string  
(album, title)               99404 non-null  string  
(artist, name)               99404 non-null  string  
(artist, website)            74276 non-null  string  
(set, split)                 99404 non-null  string  
(track, language_code)       14446 non-null  string 
(track, title)               99404 non-null  string  

"""
df["album", "engineer"] = (~df["album", "engineer"].isnull()).astype(int)
df["album", "producer"] = (~df["album", "producer"].isnull()).astype(int)
df["artist", "website"] = (~df["artist", "website"].isnull()).astype(int)
df.info()

# sistemo track, language code come fece saverio, ha senso come cosa e serve farlo prima
df["track", "language_code"] = df["track", "language_code"].fillna(
    detect(str(df["track", "title"]))
)
df["track", "language_code"] = (~df["track", "language_code"].isnull()).astype(int)

# decido di eliminare  album title, artist name,set split e track title, ovviamente sono valori unici e outliers
# DEVO ELIMINARE GLI ID SENNO' COMBINANO MACELLO
column2drop = {
    ("album", "title"),
    ("artist", "name"),
    ("set", "split"),
    ("track", "title"),
    ("album", "id"),
    ("artist", "id"),
}
df.drop(column2drop, axis=1, inplace=True)

df.info()

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

# FACCIO  IL PLOTTING
plt.figure(figsize=(15, 17))
sns.boxplot(data=df, orient="h")
plt.show()

# PLOT per trovare best est
import seaborn as sns

sns.set()
neigh = NearestNeighbors(n_neighbors=3)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()

"""APPLICO IL DBSCAN"""

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
"""
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
