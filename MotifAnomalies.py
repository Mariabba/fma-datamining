import matplotlib.pyplot as plt
import pandas as pd
from matrixprofile import *
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from music import MusicDB

musi = MusicDB()

print("{1} features for {0} tracks".format(*musi.df.shape))
print(musi.feat)

print(musi.df.info())

# scaled dataset
scaler = TimeSeriesScalerMeanVariance()
musi_scaled = pd.DataFrame(
    scaler.fit_transform(musi.df.values).reshape(
        musi.df.values.shape[0], musi.df.values.shape[1]
    )
)
musi_scaled.index = musi.df.index
print(musi_scaled.info())
print(musi_scaled.head(20))


"""
#Plot a ts with an offset
# Looking at some TS
offset = 0 #starting point
win = 100000 #dimension of window
x = musi.df.iloc[1, offset:(offset+win)]
x.plot()
plt.title("Feature with offset")
plt.show()
"""

"""
# Looking at some TS
x = musi.df.iloc[0]
x.plot()
plt.title("Time Series x")
plt.show()
"""

"""
#normalization
scaler = TimeSeriesScalerMeanVariance()
x_scaled = scaler.fit_transform(x.values.reshape(1,-1))
plt.title("Time Series x normalized")
plt.plot(np.arange(0, 2699), x_scaled.reshape(x_scaled.shape[1], x_scaled.shape[0]))
plt.show()
"""
# build mean time series rock
rock = musi_scaled.loc[musi.feat["genre"] == "Rock"]
rock_mean = rock.mean(axis=0)
print(rock_mean)
rock_mean.plot()
plt.title("Rock Mean")
plt.show()


# noise smooting
w = 50
rock_mean = ((rock_mean - rock_mean.mean()) / rock_mean.std()).rolling(window=w).mean()
plt.plot(rock_mean)
plt.title("Rock Mean Noise Smooted")
plt.show()

"""
w = 250
mp, mpi = matrixProfile.stomp(rock_mean_smooted.values, w)
plt.title("Matrix Profile Rock Mean Noise Smooted")
plt.plot(mp)
plt.show()
"""

# build matrix profile
w = 50
mp, mpi = matrixProfile.stomp(rock_mean.values, w)
plt.title("Matrix Profile Rock Mean")
plt.plot(mp)
plt.show()

# motif discovery
mo, mod = motifs.motifs(rock_mean.values, (mp, mpi), max_motifs=5, n_neighbors=3)

print(mo)
print(mod)

plt.plot(rock_mean.values)
colors = ["r", "g", "k", "b", "y"][: len(mo)]
for m, d, c in zip(mo, mod, colors):
    for i in m:
        m_shape = rock_mean.values[i : i + w]
        plt.plot(range(i, i + w), m_shape, color=c, lw=3)

plt.show()
