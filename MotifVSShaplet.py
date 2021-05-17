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

#let's plot our motif
motifs = pd.read_csv("musicmotif.csv")
print(motifs)
motifs = motifs.drop(columns=["StartPoint", "MinMPDistance", "CentroidName" ])
for n in range(3):
    motifs.iloc[n].plot()
    plt.show()