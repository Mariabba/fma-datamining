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
        round(km.inertia_, 4),  # OR km.inertia_ silhouette_score(df, km.labels_)
        round(km_dtw.inertia_, 4),  # OR km_drw.inertia_
    )


if __name__ == "__main__":
    """
    On the dataset created, compute clustering based on Euclidean/Manhattan and DTW distances and compare the results. To perform the clustering you can choose among different distance functions and clustering algorithms. Remember that you can reduce the dimensionality through approximation. Analyze the clusters and highlight similarities and differences.
    """
    musi = MusicDB()
    """
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
    """

    # Kmeans with SAX, grid search, multiprocessing
    k = 11
    x = musi.df
    # Rescale - but why?
    scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)  # Rescale time series
    ts = scaler.fit_transform(x)

    # build param collection
    param_collection = []
    for ki in range(3, k, 1):
        param_collection.append((x, ki))

    # make results
    pl_results = []
    event_steps = [int(len(param_collection) * i / 100) for i in range(100)]
    event_steps.append(1)
    event_steps = set(event_steps)

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
        with multiprocessing.Pool() as pool:
            for one_result in pool.imap_unordered(do_sax_kmeans, param_collection):
                pl_results.append(one_result)
                progress.advance(task_id)
                if int(progress._tasks[task_id].completed) in event_steps:
                    with open("data/tokens/progress.txt", "w") as f:
                        f.write(str(progress._tasks[task_id].completed))

    # make df
    dfm = pd.DataFrame(pl_results, columns=["k", "sse euclidean", "sse dtw"])

    # output results
    print(dfm.sort_values(by="sse euclidean").iloc[:20, :])
    print(dfm.sort_values(by="sse dtw").iloc[:20, :])
