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

from matrixprofile import *
import seaborn as sns

sns.set(
    rc={"figure.figsize": (18, 6)},
)
sns.set_theme(style="whitegrid")

def motifsanalysis2(ts, w, centroid_name):
    ts.plot()
    plt.title(centroid_name)
    plt.show()
    # build matrix profile
    mp, mpi = matrixProfile.stomp(ts.values, w)
    # print(min(mp))
    # print(max(mp))
    # print(np.where( mp==max(mp)))
    # print(mp)
    # print(mpi)

    x_coordinates = [np.where(mp == max(mp)), np.where(mp == min(mp))]
    y_coordinates = [max(mp), min(mp)]
    plt.title("Matrix Profile Centroid" + centroid_name)
    plt.plot(mp)
    # plt.scatter(x_coordinates, y_coordinates, c="Black")
    # plt.axhline(max(mp), color="Red", linestyle ="--", markersize=2)
    # plt.axhline(min(mp), color="Red", linestyle ="--", markersize=2)

    plt.show()

    # motif discovery
    mo, mod = motifs.motifs(ts.values, (mp, mpi), max_motifs=1, ex_zone=0, radius=2, n_neighbors=2)

    print(mo)
    print(mod)

    flag = True
    plt.plot(ts.values)
    colors = ["r", "g", "k", "b", "y"][: len(mo)]
    for m, d, c in zip(mo, mod, colors):
        for i in m:
            m_shape = ts.values[i : i + w]
            plt.plot(range(i, i + w), m_shape, color=c, lw=3)
            plt.title(centroid_name + " with Top-Motif")

    plt.show()


def motifsanalysis(ts, w, centroid_name, motif_df):
    ts.plot()
    plt.title(centroid_name)
    plt.show()
    # build matrix profile
    mp, mpi = matrixProfile.stomp(ts.values, w)
    # print(min(mp))
    # print(max(mp))
    # print(np.where( mp==max(mp)))
    # print(mp)
    # print(mpi)

    x_coordinates = [np.where(mp == max(mp)), np.where(mp == min(mp))]
    y_coordinates = [max(mp), min(mp)]
    plt.title("Matrix Profile Centroid" + centroid_name)
    plt.plot(mp)
    # plt.scatter(x_coordinates, y_coordinates, c="Black")
    # plt.axhline(max(mp), color="Red", linestyle ="--", markersize=2)
    # plt.axhline(min(mp), color="Red", linestyle ="--", markersize=2)

    plt.show()

    # motif discovery
    mo, mod = motifs.motifs(ts.values, (mp, mpi), ex_zone=0, radius=2, n_neighbors=2)

    print(mo)
    print(mod)

    flag = True
    plt.plot(ts.values)
    colors = ["r", "g", "k", "b", "y"][: len(mo)]
    for m, d, c in zip(mo, mod, colors):
        for i in m:
            # print("Starting point:", i)
            # print("Motif: ", ts.values[i:i + w])
            # print("Distance: ", d)
            row = pd.Series(ts.values[i : i + w]).append(pd.Series(i, index=[15]))
            row = row.append(pd.Series(d, index=[16]))
            row = row.append(pd.Series(centroid_name, index=[17]))
            motif_df = motif_df.append(row, ignore_index=True)
            m_shape = ts.values[i : i + w]
            plt.plot(range(i, i + w), m_shape, color=c, lw=3)
            plt.title(centroid_name + " with Top-Motif")

    plt.show()
    return motif_df


if __name__ == "__main__":

    motif_df = pd.DataFrame()
    # read dataset
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

    for n in range(8):
        centroid_name = "Centroid " + str(n)
        motif_df = motifsanalysis(
            centroids.iloc[n], 15, centroid_name=centroid_name, motif_df=motif_df
        )

    motif_df = motif_df.rename(
        columns={15: "StartPoint", 16: "MinMPDistance", 17: "CentroidName"}
    )
    motif_df = (
        motif_df.sort_values(by="MinMPDistance", ascending=True)
        .drop_duplicates(subset="MinMPDistance")
        .dropna()
    )

    motif_df = motif_df.drop(
        [
            motif_df.index[1],
            motif_df.index[3],
            motif_df.index[4],
            motif_df.index[7],
            motif_df.index[8],
            motif_df.index[12],
            motif_df.index[16],
            motif_df.index[17],
        ]
    )
    print(motif_df)

    my_colors = [
        "tab:orange",
        "blue",
        "tab:orange",
        "red",
        "purple",
        "pink",
        "purple",
        "pink",
        "tab:brown",
        "pink",
    ]
    motif_df.index = motif_df["CentroidName"]
    motif_df["MinMPDistance"].head(10).plot(
        kind="bar", x="CentroidName", y="MinMPDistance", color=my_colors
    )
    plt.title("Minimum Matrix profile value for each motif couple", fontsize=20)
    plt.ylabel("Distance", fontsize=18)
    plt.xticks(rotation=20)
    plt.show()

    motifsanalysis2(
        centroids.iloc[2], 30, centroid_name="Centroid 2")

    motifsanalysis2(
        centroids.iloc[7], 30, centroid_name="Centroid 7")


    #motif_df.head(10).to_csv("musicmotif.csv", index=False)
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
