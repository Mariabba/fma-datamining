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
cluster2 = clusters[clusters["ClusterLabel"] == 2 ]
cluster3 = clusters[clusters["ClusterLabel"] == 3 ]
cluster4 = clusters[clusters["ClusterLabel"] == 4 ]
cluster5 = clusters[clusters["ClusterLabel"] == 5 ]
cluster6 = clusters[clusters["ClusterLabel"] == 6 ]
cluster7 = clusters[clusters["ClusterLabel"] == 7 ]


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

"""
#distance dtw between motifs
for i in range(10):
    for y in range(10):
        if i != y:
            dist = dtw(motifs.iloc[i], motifs.iloc[y])
            if dist < 4:
                print("Distance between ", i, "and ", y)
                print(dist)
    print("Finished")
"""

"""
Distance between  4 and  7
3.000682959869468
Distance between  0 and  8
3.1748638275716208
Distance between  0 and  5
3.4931194805559893

"""

sns.set(
    rc={"figure.figsize": (10, 4)},
)
#plot 4/7
plt.title("DTW: 3.001")
plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.plot(motifs.iloc[4], color="purple")
plt.title("Motif from centroid 4 ")
plt.xticks([])

plt.subplot(1, 2, 2) # index 2
plt.plot(motifs.iloc[7], color="pink")
plt.title("Motif from centroid 6")
plt.xticks([])


plt.show()

#plot 0/8
plt.title("DTW: 3.175")
plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.plot(motifs.iloc[0], color="orange")
plt.title("Motif from centroid 1 ")
plt.xticks([])

plt.subplot(1, 2, 2) # index 2
plt.plot(motifs.iloc[8], color="brown")
plt.title("Motif from centroid 5")
plt.xticks([])


plt.show()

#plot 0/5
plt.title("DTW: 3.493")
plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.plot(motifs.iloc[0], color="orange")
plt.title("Motif from centroid 1 ")
plt.xticks([])

plt.subplot(1, 2, 2) # index 2
plt.plot(motifs.iloc[5], color="pink")
plt.title("Motif from centroid 6")
plt.xticks([])


plt.show()

#let's find somethig interesting between:
# 4 and 6, 1 and 5, 1 and 6
#remember that cluster 5 has the higher MinMPDistance

tracks = utils.load_tracks(
    "data/tracks.csv", outliers=False, fill=False)
print(tracks.info())

tracks0 = tracks[tracks.index.isin(cluster0.index)]
tracks1 = tracks[tracks.index.isin(cluster1.index)]
tracks2 = tracks[tracks.index.isin(cluster2.index)]
tracks3 = tracks[tracks.index.isin(cluster3.index)]
tracks4 = tracks[tracks.index.isin(cluster4.index)]
tracks5 = tracks[tracks.index.isin(cluster5.index)]
tracks6 = tracks[tracks.index.isin(cluster6.index)]
tracks7 = tracks[tracks.index.isin(cluster7.index)]

print("All dataset")
print(tracks["track", "favorites"].describe())
plt.show()

statistics = pd.DataFrame()
row0 = []
row0 = np.append(row0, round(tracks0["track", "comments"].mean(), 2))
row0 = np.append(row0, round(tracks0["track", "date_created"].mean(), 2))
row0 = np.append(row0, round(tracks0["track", "favorites"].mean(), 2))
row0 = np.append(row0, round(tracks0["track", "interest"].mean(), 2))
row0 = np.append(row0, round(tracks0["track", "listens"].mean(), 2))

statistics = statistics.append(pd.Series(row0), ignore_index=True)

row1 = []
row1 = np.append(row1, round(tracks1["track", "comments"].mean(), 2))
row1 = np.append(row1, round(tracks1["track", "date_created"].mean(), 2))
row1 = np.append(row1, round(tracks1["track", "favorites"].mean(), 2))
row1 = np.append(row1, round(tracks1["track", "interest"].mean(), 2))
row1 = np.append(row1, round(tracks1["track", "listens"].mean(), 2))

statistics = statistics.append(pd.Series(row1), ignore_index=True)

row2 = []
row2 = np.append(row2, round(tracks2["track", "comments"].mean(), 2))
row1 = np.append(row1, round(tracks1["track", "date_created"].mean(), 2))
row1 = np.append(row1, round(tracks1["track", "favorites"].mean(), 2))
row1 = np.append(row1, round(tracks1["track", "interest"].mean(), 2))
row1 = np.append(row1, round(tracks1["track", "listens"].mean(), 2))

statistics = statistics.append(pd.Series(row1), ignore_index=True)




statistics = statistics.rename(columns={0:"comments", 1:"date_created", 2:"favorites", 3:"interest", 4:"listens"})
print(statistics)