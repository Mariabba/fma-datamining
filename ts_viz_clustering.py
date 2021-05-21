import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tslearn.clustering.kmeans import TimeSeriesKMeans

from music import MusicDB


sns.set_theme(style="darkgrid")

# knee method for sse - euclidean and dtw
data = {
    10: [30.1163, 2.4920],
    9: [30.2398, 2.6395],
    8: [30.3462, 2.9231],
    7: [30.4858, 3.2075],
    6: [31.4303, 3.6130],
    5: [31.6634, 4.0992],
    4: [32.0260, 5.7189],
    3: [36.1815, 7.3733],
}

miao = pd.DataFrame.from_dict(
    data, orient="index", columns=["Euclidean distances", "DTW distances"]
)
print(miao.info())
print(miao.head(20))
sns.lineplot(x=miao.index, y="Euclidean distances", data=miao)
plt.show()

sns.lineplot(x=miao.index, y="DTW distances", data=miao)
plt.show()


# centroid characterization
"""musi = MusicDB()

km = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=5, random_state=5138)
km.fit(musi.sax)

miao = pd.DataFrame(km.cluster_centers_.squeeze())
miao.to_csv("miaomiao.csv")
"""
miao = pd.read_csv("miaomiao.csv", index_col=0)
print(miao)
print(miao.info())

w = 5
with5 = ((miao - miao.mean()) / miao.std()).rolling(window=w).mean()
plt.plot(with5.T)
plt.show()

w = 10
with10 = ((miao - miao.mean()) / miao.std()).rolling(window=w).mean()
plt.plot(with10.T)
plt.show()
