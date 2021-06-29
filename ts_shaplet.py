import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from tslearn.shapelets import ShapeletModel
from tslearn.utils import ts_size

from music import MusicDB

"""SHAPLET RETRIVE WITH ALL DATASET"""


# Carico il dataframe
musi = MusicDB()
print(musi.df.info())

X = musi.df
y = musi.feat["enc_genre"]  # classe targed ovvero genere con l'encoding

"""Creazione shaplet"""
# versione 1
n_ts, ts_sz = X.shape
n_classes = len(set(y))

# Set the number of shapelets per size as done in the original paper
# shapelet_sizes = grabocka_params_to_shapelet_size_dict(
#    n_ts=n_ts, ts_sz=ts_sz, n_classes=n_classes, l=0.1, r=1
# )
shapelet_sizes = {250: 24}

print("n_ts", n_ts)
print("ts_sz", ts_sz)
print("n_classes", n_classes)
print("shapelet_sizes", shapelet_sizes)

# Define the model using parameters provided by the authors (except that we use
# fewer iterations here)
shp_clf = ShapeletModel(
    n_shapelets_per_size=shapelet_sizes,
    optimizer="sgd",
    weight_regularizer=0.01,
    max_iter=50,
    verbose=1,
)

shp_clf.fit(X, y)

predicted_labels = shp_clf.predict(X)
print("Correct classification rate:", accuracy_score(y, predicted_labels))

predicted_locations = shp_clf.locate(X)
ts_id = 0
n_shapelets = sum(shapelet_sizes.values())

"""plot shaplet e dataset"""

sns.set(
    rc={"figure.figsize": (25, 7)},
)
plt.title(
    "Example locations of shapelet matches "
    "(24 shapelets extracted)".format(n_shapelets)
)

plt.plot(X[ts_id].ravel(), label="Time Series")
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[ts_id, idx_shp]
    plt.plot(np.arange(t0, t0 + len(shp)), shp, linewidth=0.5)
plt.legend()
plt.show()

# plot tslearn solo shaplet

# Make predictions and calculate accuracy score

# Plot the different discovered shapelets
sns.set(rc={"figure.figsize": (25, 7)})
for i, sz in enumerate(shapelet_sizes.keys()):
    plt.subplot(len(shapelet_sizes), 1, i + 1)
    plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
    for shp in shp_clf.shapelets_:
        if ts_size(shp) == sz:
            plt.plot(shp.ravel())
    plt.xlim([0, max(shapelet_sizes.keys()) - 1])

plt.tight_layout()
plt.show()

# plot singoli shaplet

sel_shapelets = np.asarray(
    [
        shp_clf.shapelets_[0],
        shp_clf.shapelets_[1],
        shp_clf.shapelets_[2],
        shp_clf.shapelets_[3],
        shp_clf.shapelets_[4],
        shp_clf.shapelets_[5],
        shp_clf.shapelets_[6],
        shp_clf.shapelets_[7],
        shp_clf.shapelets_[8],
        shp_clf.shapelets_[9],
        shp_clf.shapelets_[10],
        shp_clf.shapelets_[11],
        shp_clf.shapelets_[12],
        shp_clf.shapelets_[13],
        shp_clf.shapelets_[14],
        shp_clf.shapelets_[15],
        shp_clf.shapelets_[16],
        shp_clf.shapelets_[17],
        shp_clf.shapelets_[18],
        shp_clf.shapelets_[19],
        shp_clf.shapelets_[20],
        shp_clf.shapelets_[21],
        shp_clf.shapelets_[22],
        shp_clf.shapelets_[23],
    ],
)
sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets[0], color="blue")
axs[0, 0].set_title("shaplet 0")

axs[0, 1].plot(sel_shapelets[1], color="orange")
axs[0, 1].set_title("shaplet 1")

axs[1, 0].plot(sel_shapelets[2], color="green")
axs[1, 0].set_title("shaplet 2")

axs[1, 1].plot(sel_shapelets[3], color="red")
axs[1, 1].set_title("shaplet 3")

axs[2, 0].plot(sel_shapelets[4], color="purple")
axs[2, 0].set_title("shaplet 4")

axs[2, 1].plot(sel_shapelets[5], color="brown")
axs[2, 1].set_title("shaplet 5")

axs[3, 0].plot(sel_shapelets[6], color="pink")
axs[3, 0].set_title("shaplet 6")

axs[3, 1].plot(sel_shapelets[7], color="gray")
axs[3, 1].set_title("shaplet 7")
fig.tight_layout()
plt.show()

sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets[8], color="blue")
axs[0, 0].set_title("shaplet 8")

axs[0, 1].plot(sel_shapelets[9], color="orange")
axs[0, 1].set_title("shaplet 9")

axs[1, 0].plot(sel_shapelets[10], color="green")
axs[1, 0].set_title("shaplet 10")

axs[1, 1].plot(sel_shapelets[11], color="red")
axs[1, 1].set_title("shaplet 11")

axs[2, 0].plot(sel_shapelets[12], color="purple")
axs[2, 0].set_title("shaplet 12")

axs[2, 1].plot(sel_shapelets[13], color="brown")
axs[2, 1].set_title("shaplet 13")

axs[3, 0].plot(sel_shapelets[14], color="pink")
axs[3, 0].set_title("shaplet 14")

axs[3, 1].plot(sel_shapelets[15], color="gray")
axs[3, 1].set_title("shaplet 15")
fig.tight_layout()
plt.show()

sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets[16], color="blue")
axs[0, 0].set_title("shaplet 16")

axs[0, 1].plot(sel_shapelets[17], color="orange")
axs[0, 1].set_title("shaplet 17")

axs[1, 0].plot(sel_shapelets[18], color="green")
axs[1, 0].set_title("shaplet 18")

axs[1, 1].plot(sel_shapelets[19], color="red")
axs[1, 1].set_title("shaplet 19")

axs[2, 0].plot(sel_shapelets[20], color="purple")
axs[2, 0].set_title("shaplet 20")

axs[2, 1].plot(sel_shapelets[21], color="brown")
axs[2, 1].set_title("shaplet 21")

axs[3, 0].plot(sel_shapelets[22], color="pink")
axs[3, 0].set_title("shaplet 22")

axs[3, 1].plot(sel_shapelets[23], color="gray")
axs[3, 1].set_title("shaplet 23")
fig.tight_layout()
plt.show()
