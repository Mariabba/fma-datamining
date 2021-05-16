import librosa
import pandas as pd

from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import missingno as mso
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from music import MusicDB
import IPython.display as ipd
import numpy as np
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import StandardScaler

from matrixprofile import *
import seaborn as sns

sns.set(
    rc={"figure.figsize": (18, 6)},
)
sns.set_theme(style="whitegrid")

def motifsanalysis(ts, w):
    # build matrix profile
    mp, mpi = matrixProfile.stomp(ts.values, w)
    plt.title("Matrix Profile Rock Mean")
    plt.plot(mp)
    plt.show()

    # motif discovery
    mo, mod = motifs.motifs(ts.values, (mp, mpi), max_motifs=5, n_neighbors=3)

    print(mo)
    print(mod)

    plt.plot(ts.values)
    colors = ['r', 'g', 'k', 'b', 'y'][:len(mo)]
    for m, d, c in zip(mo, mod, colors):
        for i in m:
            m_shape = ts.values[i:i + w]
            plt.plot(range(i, i + w), m_shape, color=c, lw=3)

    plt.show()

if __name__ == "__main__":

    #read dataset
    centroids = pd.read_csv("centroidiclusters_dtw.csv")

    print(centroids.info())
    print(centroids)
    centroids.T.plot()
    plt.title("Centroids")
    plt.show()

    sns.set()
    fig, axs = plt.subplots(4, 2, figsize=(10, 12))
    axs[0, 0].plot(centroids.iloc[0], color="blue")
    axs[0, 0].set_title("Cluster 0")
    axs[0, 0].set(xticklabels=[])

    axs[0, 1].plot(centroids.iloc[1], color="orange")
    axs[0, 1].set_title("Cluster 1")
    axs[0, 1].set(xticklabels=[])

    axs[1, 0].plot(centroids.iloc[2], color="green")
    axs[1, 0].set_title("Cluster 2")
    axs[1, 0].set(xticklabels=[])

    axs[1, 1].plot(centroids.iloc[3], color="red")
    axs[1, 1].set_title("Cluster 3")
    axs[1, 1].set(xticklabels=[])

    axs[2, 0].plot(centroids.iloc[4], color="purple")
    axs[2, 0].set_title("Cluster 4")
    axs[2, 0].set(xticklabels=[])

    axs[2, 1].plot(centroids.iloc[5], color="brown")
    axs[2, 1].set_title("Cluster 5")
    axs[2, 1].set(xticklabels=[])

    axs[3, 0].plot(centroids.iloc[6], color="pink")
    axs[3, 0].set_title("Cluster 6")
    axs[3, 0].set(xticklabels=[])

    axs[3, 1].plot(centroids.iloc[7], color="gray")
    axs[3, 1].set_title("Cluster 7")
    axs[3, 1].set(xticklabels=[])

    fig.tight_layout()
    plt.show()
    """
    #scaled dataset
    scaler = TimeSeriesScalerMeanVariance()
    musi_scaled = pd.DataFrame(scaler.fit_transform(musi.df.values).reshape(musi.df.values.shape[0], musi.df.values.shape[1]))
    musi_scaled.index = musi.df.index
    print(musi_scaled.info())
    print(musi_scaled.head(20))


    #build mean time series rock
    rock = musi_scaled.loc[musi.feat["genre"] == "Rock"]
    rock_mean = rock.mean(axis=0)
    print(rock_mean)
    rock_mean.plot()
    plt.title("Rock Mean")
    plt.show()


    #noise smooting
    w = 50
    rock_mean = ((rock_mean - rock_mean.mean())/rock_mean.std()).rolling(window=w).mean()
    plt.plot(rock_mean)
    plt.title("Rock Mean Noise Smooted")
    plt.show()
    motifsanalysis(rock_mean, 50)
    """

