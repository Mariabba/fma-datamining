import multiprocessing

import pandas as pd
import numpy as np
from rich import print
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from scipy.spatial.distance import cdist
from tslearn.metrics import dtw_path
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from music import MusicDB


def do_sax_knn(params):
    # unpack
    Xi, yi, segments, symbols, k = params

    # Make sax
    sax = SymbolicAggregateApproximation(n_segments=segments, alphabet_size_avg=symbols)
    ts_sax = sax.fit_transform(Xi)

    x_train, x_test, y_train, y_test = train_test_split(
        ts_sax, yi, test_size=0.2, random_state=100, stratify=yi
    )
    kn = KNeighborsClassifier(n_neighbors=k, p=1, weights="distance")
    kn.fit(np.squeeze(x_train), y_train)
    kn_d = KNeighborsClassifier(n_neighbors=k, p=2, weights="distance")
    kn_d.fit(np.squeeze(x_train), y_train)

    y_pred = kn.predict(np.squeeze(x_test))
    y_pred_manh = kn_d.predict(np.squeeze(x_test))

    return (
        segments,
        symbols,
        k,
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_manh),
    )


if __name__ == "__main__":
    musi = MusicDB()

    path, dist = dtw_path(musi.df.iloc[0], musi.df.iloc[1])

    cost_matrix = cdist(
        musi.df.iloc[0].values[:100].reshape(-1, 1),
        musi.df.iloc[1].values[:100].reshape(-1, 1),
    )

    # KNN with SAX, grid search, multiprocessing
    segments = 1000
    symbols = 100
    k = 30

    x = musi.df
    # Rescale - but why?
    scaler = TimeSeriesScalerMeanVariance()  # Rescale time series
    X = scaler.fit_transform(x)
    X = pd.DataFrame(X.reshape(musi.df.values.shape[0], musi.df.values.shape[1]))
    X.index = musi.df.index
    y = musi.feat["enc_genre"]

    # make results
    pl_results = []
    num_errors = 0

    # build param collection
    param_collection = []
    for seg in range(5, segments, 25):
        for symb in range(5, symbols, 5):
            for ki in range(3, k, 1):
                param_collection.append((X, y, seg, symb, ki))

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
                pl_eucl,
                pl_manh,
            ) in pool.imap_unordered(do_sax_knn, param_collection):
                if type(pl_eucl) is not bool:
                    resuuuu = (
                        pl_segments,
                        pl_symbols,
                        pl_k,
                        round(pl_eucl, 4),
                        round(pl_manh, 4),
                    )
                    pl_results.append(resuuuu)
                else:
                    num_errors += 1
                progress.advance(task_id)

    dfm = pd.DataFrame(
        pl_results, columns=["segments", "symbols", "k", "f1 euclidean", "f1 manhattan"]
    )
    dfm = dfm.sort_values(by="f1 euclidean", ascending=False)
    print(dfm.iloc[:20, :])
    dfm = dfm.sort_values(by="f1 manhattan", ascending=False)
    print(dfm.iloc[:20, :])
