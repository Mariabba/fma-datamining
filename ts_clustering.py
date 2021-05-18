import multiprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from rich import print
from rich.progress import BarColumn, Progress, TimeRemainingColumn
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from music import MusicDB

mpl.rcParams["figure.dpi"] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")


def do_sax_kmeans(params):
    df, k = params

    # Make sax
    sax = SymbolicAggregateApproximation(n_segments=130, alphabet_size_avg=20)
    ts_sax = sax.fit_transform(df)
    sax_dataset_inv = sax.inverse_transform(ts_sax)

    km_dtw = TimeSeriesKMeans(
        n_clusters=k, metric="euclidean", max_iter=50, random_state=5138
    )
    km_dtw.fit(ts_sax)
    """
    km_dtw = TimeSeriesKMeans(
        n_clusters=k, metric="dtw", max_iter=50, random_state=5138
    )
    km_dtw.fit(ts_sax)
    """
    return (
        km_dtw.cluster_centers_,
        km_dtw.labels_,
        round(km_dtw.inertia_, 4),
    )


if __name__ == "__main__":
    """
    On the dataset created, compute clustering based on Euclidean/Manhattan and DTW distances and compare the results. To perform the clustering you can choose among different distance functions and clustering algorithms. Remember that you can reduce the dimensionality through approximation. Analyze the clusters and highlight similarities and differences.
    """
    musi = MusicDB()

    # Kmeans with SAX, grid search, multiprocessing
    k = 11
    x = musi.df
    # Rescale - but why?
    scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)  # Rescale time series
    ts = scaler.fit_transform(x)

    # param_collection.append((x, 4)) to do

    """
    # populate results
    for one_result in param_collection:
        pl_results.append(do_sax_kmeans(one_result))
    """

    centroids, labels, inertia = do_sax_kmeans((ts, 8))

    musi.feat["ClusterLabel"] = labels
    musi.feat = musi.feat.drop(["enc_genre"], axis=1)

    plt.plot(np.squeeze(centroids).T)
    plt.show()
    df_centroids = pd.DataFrame()
    df_centroids = df_centroids.append(pd.Series(centroids[0, :, 0]), ignore_index=True)
    df_centroids = df_centroids.append(pd.Series(centroids[1, :, 0]), ignore_index=True)
    df_centroids = df_centroids.append(pd.Series(centroids[2, :, 0]), ignore_index=True)
    df_centroids = df_centroids.append(pd.Series(centroids[3, :, 0]), ignore_index=True)
    df_centroids = df_centroids.append(pd.Series(centroids[4, :, 0]), ignore_index=True)
    df_centroids = df_centroids.append(pd.Series(centroids[5, :, 0]), ignore_index=True)
    df_centroids = df_centroids.append(pd.Series(centroids[6, :, 0]), ignore_index=True)
    df_centroids = df_centroids.append(pd.Series(centroids[7, :, 0]), ignore_index=True)

    print(df_centroids)
    print(musi.feat)

    musi.feat = musi.feat.groupby(["genre", "ClusterLabel"], as_index=False).size()
    musi.feat = musi.feat[musi.feat["size"] != 0]
    musi.feat = musi.feat.sort_values(by=["ClusterLabel"])

    df_centroids.to_csv("centroidiclusters.csv", index=False)
    musi.feat.to_csv("musicluster.csv", index=False)
