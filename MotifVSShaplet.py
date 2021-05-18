import librosa
import pandas as pd

from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import missingno as mso
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import argrelextrema
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from music import MusicDB
import IPython.display as ipd
import numpy as np
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import utils
from tslearn.metrics import dtw, dtw_path, cdist_dtw, subsequence_cost_matrix

clusters = pd.read_csv("musicluster_dtw_index.csv", index_col="Unnamed: 0")
cluster0 = clusters[clusters["ClusterLabel"] == 0 ]
cluster1 = clusters[clusters["ClusterLabel"] == 1 ]
cluster3 = clusters[clusters["ClusterLabel"] == 3 ]
cluster4 = clusters[clusters["ClusterLabel"] == 4 ]
cluster6 = clusters[clusters["ClusterLabel"] == 6 ]


tracks = utils.load_tracks(
    "data/tracks.csv", outliers=False, fill=False)
print(tracks.info())

tracks0 = tracks[tracks.index.isin(cluster0.index)]
tracks1 = tracks[tracks.index.isin(cluster1.index)]
tracks3 = tracks[tracks.index.isin(cluster3.index)]
tracks4 = tracks[tracks.index.isin(cluster4.index)]
tracks6 = tracks[tracks.index.isin(cluster6.index)]

print(tracks["track", "favorites"].hist())
plt.show()

print(tracks0["track", "favorites"].hist())
plt.show()

print(tracks1["track", "favorites"].hist())
plt.show()

print(tracks3["track", "favorites"].hist())
plt.show()

print(tracks4["track", "favorites"].hist()) #track, comments #track, favorites
plt.show()

print(tracks6["track", "favorites"].hist()) #album,comments
plt.show()

#let's plot our motif
motifs = pd.read_csv("musicmotif.csv")
print(motifs)

motifs = motifs.drop(columns=["StartPoint", "MinMPDistance", "CentroidName" ])

sns.set()
fig, axs = plt.subplots(5, 2, figsize=(10, 12))

axs[0, 0].plot(motifs.iloc[0], color="orange")
axs[0, 0].set_title("Cluster 1")
axs[0, 0].set(xticklabels=[])

axs[0, 1].plot(motifs.iloc[1], color="blue")
axs[0, 1].set_title("Cluster 0")
axs[0, 1].set(xticklabels=[])

axs[1, 0].plot(motifs.iloc[2], color="orange")
axs[1, 0].set_title("Cluster 1")
axs[1, 0].set(xticklabels=[])

axs[1, 1].plot(motifs.iloc[3], color="red")
axs[1, 1].set_title("Cluster 3")
axs[1, 1].set(xticklabels=[])

axs[2, 0].plot(motifs.iloc[4], color="purple")
axs[2, 0].set_title("Cluster 4")
axs[2, 0].set(xticklabels=[])

axs[2, 1].plot(motifs.iloc[5], color="pink")
axs[2, 1].set_title("Cluster 6")
axs[2, 1].set(xticklabels=[])

axs[3, 0].plot(motifs.iloc[6], color="purple")
axs[3, 0].set_title("Cluster 4")
axs[3, 0].set(xticklabels=[])

axs[3, 1].plot(motifs.iloc[7], color="pink")
axs[3, 1].set_title("Cluster 6")
axs[3, 1].set(xticklabels=[])

axs[4, 0].plot(motifs.iloc[8], color="brown")
axs[4, 0].set_title("Cluster 5")
axs[4, 0].set(xticklabels=[])

axs[4, 1].plot(motifs.iloc[9], color="pink")
axs[4, 1].set_title("Cluster 6")
axs[4, 1].set(xticklabels=[])

fig.tight_layout()
plt.show()

for i in range(10):
    for y in range(10):
        if i != y:
            print("Distance between ", i, "and ", y)
            dist = dtw(motifs.iloc[i], motifs.iloc[y])
            print(dist)
    print("Finished")

"""
plot two figure
plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.plot(motifs.iloc[7])
plt.title("My first plot!")
plt.xlabel('X-axis ')
plt.ylabel('Y-axis ')

plt.subplot(1, 2, 2) # index 2
plt.plot(motifs.iloc[5])
plt.title("My second plot!")
plt.xlabel('X-axis ')
plt.ylabel('Y-axis ')

plt.show()
"""