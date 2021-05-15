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
    rc={"figure.figsize": (10, 6)},
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


#read dataset
musi = MusicDB()

print("{1} features for {0} tracks".format(*musi.df.shape))
print(musi.feat)

print(musi.df.info())

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

