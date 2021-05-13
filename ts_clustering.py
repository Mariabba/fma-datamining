import multiprocessing
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from rich import print
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from scipy.spatial.distance import cdist
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.metrics import dtw_path
from tslearn.piecewise import (
    PiecewiseAggregateApproximation,
    SymbolicAggregateApproximation,
)
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from music import MusicDB

mpl.rcParams["figure.dpi"] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")


def do_sax_kmeans(params):
    plots = False
    # unpack
    df, segments, symbols, k = params

    # Make sax
    sax = SymbolicAggregateApproximation(n_segments=segments, alphabet_size_avg=symbols)
    ts_sax = sax.fit_transform(df)
    sax_dataset_inv = sax.inverse_transform(ts_sax)

    if plots:
        plt.plot(df[0].ravel(), "b-", alpha=0.4)
        plt.plot(sax_dataset_inv[0].ravel(), "b-")
        plt.title("SAX, %d symbols" % symbols)
        plt.show()

    km = TimeSeriesKMeans(
        n_clusters=k, metric="euclidean", max_iter=50, random_state=5138
    )
    km.fit(ts_sax)
    # print(silhouette_score(df, km.labels_))  # this takes ages
    km_d = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=50, random_state=5138)
    km_d.fit(ts_sax)

    if plots:
        # centroids
        plt.plot(km.cluster_centers_.reshape(ts_sax.shape[1], k))
        plt.show()
        # WHAT THE f IS THIS
        for i in range(k):
            plt.plot(np.mean(df[np.where(km.labels_ == i)[0]], axis=0))
        plt.show()
    return segments, symbols, k, km.inertia_, km_d.inertia_


if __name__ == "__main__":
    """
    On the dataset created, compute clustering based on Euclidean/Manhattan and DTW distances and compare the results. To perform the clustering you can choose among different distance functions and clustering algorithms. Remember that you can reduce the dimensionality through approximation. Analyze the clusters and highlight similarities and differences.
    """
    try:
        method_chosen = int(sys.argv[1])
    except IndexError:
        raise SystemExit("Remember to supply a parameter.")

    musi = MusicDB()

    path, dist = dtw_path(musi.df.iloc[0], musi.df.iloc[1])

    cost_matrix = cdist(
        musi.df.iloc[0].values[:100].reshape(-1, 1),
        musi.df.iloc[1].values[:100].reshape(-1, 1),
    )

    if method_chosen == 1:
        # pathing
        fig, ax = plt.subplots(figsize=(12, 8))
        ax = sbn.heatmap(
            cost_matrix, annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax
        )
        ax.invert_yaxis()
        # Get the warp path in x and y directions
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        # Align the path from the center of each cell
        path_xx = [x + 0.5 for x in path_x]
        path_yy = [y + 0.5 for y in path_y]
        ax.plot(path_xx, path_yy, color="blue", linewidth=3, alpha=0.2)
        fig.savefig("ex1_heatmap.png", **savefig_options)

    elif method_chosen == 2:
        # Kmeans
        km = TimeSeriesKMeans(
            n_clusters=8, metric="euclidean", max_iter=5, random_state=1149
        )
        km.fit(musi.df.iloc[:, :200])
        print(km.cluster_centers_.shape)
        plt.plot(np.squeeze(km.cluster_centers_).T)
        plt.show()
        print(km.inertia_)

    elif method_chosen == 3:
        # KMeans with PAA approximation
        segments = 10
        n_clusters = 8
        percent_rows_to_use = 1

        rows_to_use = round(len(musi.df) * percent_rows_to_use / 100)
        x = musi.df.iloc[:rows_to_use, :]
        paa = PiecewiseAggregateApproximation(n_segments=segments)
        X_paa = paa.fit_transform(x)
        plt.plot(X_paa.reshape(X_paa.shape[1], X_paa.shape[0]))
        plt.show()

        km = TimeSeriesKMeans(
            n_clusters=n_clusters, metric="euclidean", max_iter=5, random_state=0
        )
        km.fit(X_paa)
        # plt.plot(km.cluster_centers_.reshape(X_paa.shape[1], n_clusters))
        # plt.show()

        for i in range(n_clusters):
            plt.plot(np.mean(x[np.where(km.labels_ == i)[0]], axis=0))
        plt.show()

    elif method_chosen == 4:
        # Kmeans with SAX, grid search, multiprocessing
        segments = 50
        symbols = 25
        k = 10
        # percent_rows_to_use = 100

        # rows_to_use = round(len(musi.df) * percent_rows_to_use / 100)
        x = musi.df  # .iloc[:rows_to_use, :]
        # Rescale - but why?
        scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)  # Rescale time series
        ts = scaler.fit_transform(x)

        # make results
        pl_results = []
        num_errors = 0

        # build param collection
        param_collection = []
        for seg in range(5, segments, 5):
            for symb in range(5, symbols, 5):
                for ki in range(4, k, 1):
                    param_collection.append((x, seg, symb, ki))

        # make progress reporting
        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "{task.completed} of {task.total}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        )

        # populate df
        with progress:
            task_id = progress.add_task("[cyan]KMeans…", total=len(param_collection))
            with multiprocessing.Pool() as pool:
                for (
                    pl_segments,
                    pl_symbols,
                    pl_k,
                    pl_sse,
                    pl_eps,
                ) in pool.imap_unordered(do_sax_kmeans, param_collection):
                    if type(pl_eps) is not bool:
                        pl_results.append(
                            (
                                pl_segments,
                                pl_symbols,
                                pl_k,
                                round(pl_sse, 4),
                                round(pl_eps, 4),
                            )
                        )
                    else:
                        num_errors += 1
                    progress.advance(task_id)

        dfm = pd.DataFrame(
            pl_results, columns=["segments", "symbols", "k", "sse euclidean", "sse dtw"]
        )
        dfm = dfm.sort_values(by="sse euclidean")
        print(dfm.iloc[:20, :])
        dfm = dfm.sort_values(by="sse dtw")
        print(dfm.iloc[:20, :])
