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
    plots = False
    # unpack
    df, k = params

    # Make sax
    sax = SymbolicAggregateApproximation(n_segments=130, alphabet_size_avg=20)
    ts_sax = sax.fit_transform(df)
    sax_dataset_inv = sax.inverse_transform(ts_sax)

    km = TimeSeriesKMeans(
        n_clusters=k, metric="euclidean", max_iter=50, random_state=5138
    )
    km.fit(ts_sax)

    km_dtw = TimeSeriesKMeans(
        n_clusters=k, metric="dtw", max_iter=50, random_state=5138
    )
    km_dtw.fit(ts_sax)

    if plots:
        # centroids
        plt.plot(km.cluster_centers_.reshape(ts_sax.shape[1], k))
        plt.show()
        # WHAT THE f IS THIS
        for i in range(k):
            plt.plot(np.mean(df[np.where(km.labels_ == i)[0]], axis=0))
        plt.show()
    return (
        k,
        round(km.inertia_, 4),
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

    # build param collection
    param_collection = [(x, 8)]
    #param_collection.append((x, 4)) to do

    # make results
    pl_results = []

    # make progress reporting
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "{task.completed} of {task.total}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )

    # populate results
    with progress:
        task_id = progress.add_task("[cyan]KMeans…", total=len(param_collection))
        #with multiprocessing.Pool() as pool:
        for one_result in param_collection:
            pl_results.append(do_sax_kmeans(one_result))
            progress.advance(task_id)

    # make df
    dfm = pd.DataFrame(pl_results, columns=["k", "sse euclidean", "sse dtw"])

    # output results
    print(dfm.sort_values(by="sse euclidean").iloc[:20, :])
    print(dfm.sort_values(by="sse dtw").iloc[:20, :])
