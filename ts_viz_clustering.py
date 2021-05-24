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
"""print(miao.info())
print(miao.head(20))
sns.lineplot(x=miao.index, y="Euclidean distances", data=miao)
plt.show()

sns.lineplot(x=miao.index, y="DTW distances", data=miao)
plt.show()
"""

# centroid characterization
musi = MusicDB()

km = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=5, random_state=5138)
km.fit(musi.sax)

miao = pd.DataFrame(km.cluster_centers_.squeeze())
# miao.to_csv("miaomiao.csv")

# miao = pd.read_csv("miaomiao.csv", index_col=0)

w = 20
for i in range(5):
    with5 = (
        ((miao.iloc[i] - miao.iloc[i].mean()) / miao.iloc[i].std())
        .rolling(window=w)
        .mean()
    )
    plt.plot(with5)
plt.show()

musi.feat["ClusterLabel"] = km.labels_
musi.feat = musi.feat.drop(["enc_genre"], axis=1)

musi.feat = musi.feat.groupby(["genre", "ClusterLabel"], as_index=False).size()
musi.feat = musi.feat[musi.feat["size"] != 0]
musi.feat = musi.feat.sort_values(by=["ClusterLabel"])
musi.feat.index = musi.feat["genre"]

cluster1 = musi.feat[musi.feat["ClusterLabel"] == 1].sort_values(by=["size"])
cluster1["size"].plot(kind="bar", x="genre")
plt.title("Tracks genre distribution - cluster 1")
plt.xticks(rotation=18)
plt.show()

cluster4 = musi.feat[musi.feat["ClusterLabel"] == 2].sort_values(by=["size"])
cluster4["size"].plot(kind="bar", x="genre")
plt.title("Tracks genre distribution - cluster 2")
plt.xticks(rotation=18)
plt.show()

cluster5 = musi.feat[musi.feat["ClusterLabel"] == 3].sort_values(by=["size"])
cluster5["size"].plot(kind="bar", x="genre")
plt.title("Tracks genre distribution - cluster 3")
plt.xticks(rotation=18)
plt.show()

cluster6 = musi.feat[musi.feat["ClusterLabel"] == 4].sort_values(by=["size"])
cluster6["size"].plot(kind="bar", x="genre")
plt.title("Tracks genre distribution - cluster 4")
plt.xticks(rotation=18)
plt.show()

cluster1 = musi.feat[musi.feat["ClusterLabel"] == 5].sort_values(by=["size"])
cluster1["size"].plot(kind="bar", x="genre")
plt.title("Tracks genre distribution - cluster 5")
plt.xticks(rotation=18)
plt.show()
