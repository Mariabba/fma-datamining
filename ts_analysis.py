# pandas libraries
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import missingno as mso
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from music import MusicDB

# Carico musi come dataframe 62 rows
musi = MusicDB()
musi.df.info()
print(musi.df)

print("{1} features for {0} tracks".format(*musi.df.shape))
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

# Visualize con seaborn

sns.set(rc={"figure.figsize": (11, 4)})
plt.plot(musi.df, linewidth=0.5)
plt.show()

# sono pazza scusate
from tslearn.clustering import TimeSeriesKMeans

# from tslearn.generators import random_walks
"""
X = random_walks(n_ts=50, sz=32, d=1)
X.shape
np.squeeze(X).shape
plt.plot(np.squeeze(X).T)
plt.show()
"""
