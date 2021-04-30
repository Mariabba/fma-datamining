# pandas libraries
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

# Carico musi come dataframe 62 rows


musi = MusicDB()
musi.df.info()
print(musi.df)

print("{1} features for {0} tracks".format(*musi.df.shape))
"""
# First plot of musi
plt.plot(musi.df)
plt.title("Music small 62 features")
plt.show()

# Looking at some TS

x = musi.df[139]
y = musi.df[2]
z = musi.df[5]
x.plot()
y.plot()
z.plot()
plt.title("Some features")
plt.show()
"""
# Visualize con seaborn

sns.set(rc={"figure.figsize": (11, 4)})
plt.plot(musi.df, linewidth=0.5)
plt.show()
"""
# Ascoltare le canzoni

filename = musi.df[2]
print("File: {}".format(filename))

x, sr = librosa.load(filename, sr=None, mono=True)
print("Duration: {:.2f}s, {} samples".format(x.shape[-1] / sr, x.size))

start, end = 7, 17
ipd.Audio(data=x[start * sr : end * sr], rate=sr)
"""  # ESTRAZIONE FEATURES
"""


def calculate_features(values):
    features = {
        "avg": np.mean(values),
        "std": np.std(values),
        "var": np.var(values),
        "med": np.median(values),
        "10p": np.percentile(values, 10),
        "25p": np.percentile(values, 25),
        "50p": np.percentile(values, 50),
        "75p": np.percentile(values, 75),
        "90p": np.percentile(values, 90),
        "iqr": np.percentile(values, 75) - np.percentile(values, 25),
        "cov": 1.0 * np.mean(values) / np.std(values),
        "skw": stats.skew(values),
        "kur": stats.kurtosis(values),
    }

    return features


features = calculate_features(musi.df)
print("Features", features)
"""
from tsfresh.feature_extraction import extract_features

"""Time series Approximation"""
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.piecewise import OneD_SymbolicAggregateApproximation

scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)  # Rescale time series
ts = scaler.fit_transform(musi.df.values.reshape(1, -1))

# PAA transform (and inverse transform) of the data
n_paa_segments = 50
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
ts_paa = paa.fit_transform(ts)
paa_dataset_inv = paa.inverse_transform(ts_paa)

# SAX transform
n_sax_symbols = 50
sax = SymbolicAggregateApproximation(
    n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols
)
ts_sax = sax.fit_transform(ts)
sax_dataset_inv = sax.inverse_transform(ts_sax)

# 1d-SAX transform
n_sax_symbols_avg = 100
n_sax_symbols_slope = 40
one_d_sax = OneD_SymbolicAggregateApproximation(
    n_segments=n_paa_segments,
    alphabet_size_avg=n_sax_symbols_avg,
    alphabet_size_slope=n_sax_symbols_slope,
)

ts_sax1d = one_d_sax.fit_transform(ts)
one_d_sax_dataset_inv = one_d_sax.inverse_transform(ts_sax1d)
"""
plt.figure()
plt.subplot(2, 2, 1)  # First, raw time series
plt.plot(ts[0].ravel(), "b-")
plt.title("Raw time series")

plt.subplot(2, 2, 2)  # Second, PAA
plt.plot(ts[0].ravel(), "b-", alpha=0.4)
plt.plot(paa_dataset_inv[0].ravel(), "b-")
plt.title("PAA")

plt.subplot(2, 2, 3)  # Then SAX
plt.plot(ts[0].ravel(), "b-", alpha=0.4)
plt.plot(sax_dataset_inv[0].ravel(), "b-")
plt.title("SAX, %d symbols" % n_sax_symbols)

plt.subplot(2, 2, 4)  # Finally, 1d-SAX
plt.plot(ts[0].ravel(), "b-", alpha=0.4)
plt.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
plt.title(
    "1d-SAX, %d symbols"
    "(%dx%d)"
    % (n_sax_symbols_avg * n_sax_symbols_slope, n_sax_symbols_avg, n_sax_symbols_slope)
)

plt.tight_layout()
plt.show()
"""

plt.plot(ts[0].ravel())
plt.title("normale dataset")
plt.show()

"""
ts1_paa = paa.fit_transform(ts)
plt.plot(paa.inverse_transform(ts1_paa)[0].ravel())
plt.title("PAA inverse")
plt.show()
"""
ts1_sax = sax.fit_transform(ts)
plt.plot(sax.inverse_transform(ts1_sax)[0].ravel())
plt.title("Saxxx")
plt.show()

"""
ts1_sax1d = one_d_sax.fit_transform(ts)
plt.plot(one_d_sax.inverse_transform(ts1_sax1d)[0].ravel())
plt.title("one-d saxxx")
plt.show()
"""

"""K-Means with Sax"""

km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0)
km.fit(ts1_sax)

plt.plot(km.cluster_centers_.reshape(ts1_sax.shape[1], 3))
plt.title("Cluster with k=3 and sax approximation")
plt.show()

hist, bins = np.histogram(km.labels_, bins=range(0, len(set(km.labels_)) + 1))
print("centers", km.cluster_centers_.shape)
plt.plot(np.squeeze(km.cluster_centers_).T)
plt.show()
print()
print("Labels: ", km.labels_)
print()
print("SSE: ", km.inertia_)
