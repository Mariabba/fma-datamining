import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from music import MusicDB

from tslearn.metrics import dtw

from scipy.spatial.distance import euclidean

musi = MusicDB()
print(musi.df.info())

X = musi.df
y = musi.feat["enc_genre"]

print(musi.feat["genre"].unique())

sel_shapelets = pd.read_csv("musicshaplet.csv")

motif_df = pd.read_csv("musicmotif.csv")
motif_df = motif_df.drop(columns=["StartPoint", "MinMPDistance", "CentroidName"])
"""
sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets.iloc[0, :], color="blue")
axs[0, 0].set_title("shapelet 0")

axs[0, 1].plot(sel_shapelets.iloc[1, :], color="orange")
axs[0, 1].set_title("shapelet 1")

axs[1, 0].plot(sel_shapelets.iloc[2, :], color="green")
axs[1, 0].set_title("shapelet 2")

axs[1, 1].plot(sel_shapelets.iloc[3, :], color="red")
axs[1, 1].set_title("shapelet 3")

axs[2, 0].plot(sel_shapelets.iloc[4, :], color="purple")
axs[2, 0].set_title("shapelet 4")

axs[2, 1].plot(sel_shapelets.iloc[5, :], color="brown")
axs[2, 1].set_title("shapelet 5")

axs[3, 0].plot(sel_shapelets.iloc[6, :], color="pink")
axs[3, 0].set_title("shapelet 6")

axs[3, 1].plot(sel_shapelets.iloc[7, :], color="gray")
axs[3, 1].set_title("shapelet 7")
fig.tight_layout()
plt.show()

sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets.iloc[8, :], color="blue")
axs[0, 0].set_title("shapelet 8")

axs[0, 1].plot(sel_shapelets.iloc[9, :], color="orange")
axs[0, 1].set_title("shapelet 9")

axs[1, 0].plot(sel_shapelets.iloc[10, :], color="green")
axs[1, 0].set_title("shapelet 10")

axs[1, 1].plot(sel_shapelets.iloc[11, :], color="red")
axs[1, 1].set_title("shapelet 11")

axs[2, 0].plot(sel_shapelets.iloc[12, :], color="purple")
axs[2, 0].set_title("shapelet 12")

axs[2, 1].plot(sel_shapelets.iloc[13, :], color="brown")
axs[2, 1].set_title("shapelet 13")

axs[3, 0].plot(sel_shapelets.iloc[14, :], color="pink")
axs[3, 0].set_title("shapelet 14")

axs[3, 1].plot(sel_shapelets.iloc[15, :], color="gray")
axs[3, 1].set_title("shapelet 15")
fig.tight_layout()
plt.show()

sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets.iloc[16, :], color="blue")
axs[0, 0].set_title("shapelet 16")

axs[0, 1].plot(sel_shapelets.iloc[17, :], color="orange")
axs[0, 1].set_title("shapelet 17")

axs[1, 0].plot(sel_shapelets.iloc[18, :], color="green")
axs[1, 0].set_title("shapelet 18")

axs[1, 1].plot(sel_shapelets.iloc[19, :], color="red")
axs[1, 1].set_title("shapelet 19")

axs[2, 0].plot(sel_shapelets.iloc[20, :], color="purple")
axs[2, 0].set_title("shapelet 20")

axs[2, 1].plot(sel_shapelets.iloc[21, :], color="brown")
axs[2, 1].set_title("shapelet 21")

axs[3, 0].plot(sel_shapelets.iloc[22, :], color="pink")
axs[3, 0].set_title("shapelet 22")

axs[3, 1].plot(sel_shapelets.iloc[23, :], color="gray")
axs[3, 1].set_title("shapelet 23")
fig.tight_layout()
plt.show()
"""
count = -1
print("Iniziamo dtw")
dist = []
shap = []
mot = []

for i in range(10):
    for y in range(24):
        if i != y:
            count = count + 1
            dist.append(dtw(motif_df.iloc[i], sel_shapelets.iloc[y]))
            shap.append(str(y))
            mot.append(str(i))
    print("Finished")
print(dist)

dio = [" / ".join(item) for item in zip(mot, shap)]
print("lo zip", dio)

bestemmia = pd.DataFrame(data={"Motif/Shapelet": dio, "DTW": dist}, index=dio)
print(bestemmia.info())

bestemmia = bestemmia.sort_values(by="DTW")

print(bestemmia.head(10))

sns.set()
bestemmia["DTW"].head(10).plot(
    kind="bar",
    color=[
        "red",
        "orange",
        "yellow",
        "mediumseagreen",
        "green",
        "xkcd:sky blue",
        "blue",
        "purple",
        "m",
        "pink",
    ],
    fontsize=9,
)
plt.title("Minimum DTW distance between Motif and Shapelet", fontsize=13)
plt.xticks(rotation=20)
plt.xlabel("Motif/Shapelet")
plt.ylabel("DTW distance")
plt.show()

"""PLot shaplet accanto a motif"""
sns.set()
fig, axs = plt.subplots(4, 2, figsize=(8, 6))
axs[0, 0].plot(motif_df.iloc[5, :], color="red")
axs[0, 0].set_title("motif 5")

axs[0, 1].plot(sel_shapelets.iloc[0, :], color="red")
axs[0, 1].set_title("shapelet 0")

axs[1, 0].plot(motif_df.iloc[5, :], color="orange")
axs[1, 0].set_title("motif 5")

axs[1, 1].plot(sel_shapelets.iloc[15, :], color="orange")
axs[1, 1].set_title("shapelet 15")

axs[2, 0].plot(motif_df.iloc[5, :], color="yellow")
axs[2, 0].set_title("motif 5")

axs[2, 1].plot(sel_shapelets.iloc[1, :], color="yellow")
axs[2, 1].set_title("shapelet 1")

axs[3, 0].plot(motif_df.iloc[2, :], color="mediumseagreen")
axs[3, 0].set_title("motif 2")

axs[3, 1].plot(sel_shapelets.iloc[15, :], color="mediumseagreen")
axs[3, 1].set_title("shapelet 15")

fig.tight_layout()
plt.show()


sns.set()
fig, axs = plt.subplots(4, 2, figsize=(8, 6))
axs[0, 0].plot(motif_df.iloc[2, :], color="green")
axs[0, 0].set_title("motif 2")

axs[0, 1].plot(sel_shapelets.iloc[5, :], color="green")
axs[0, 1].set_title("shapelet 5")

axs[1, 0].plot(motif_df.iloc[5, :], color="xkcd:sky blue")
axs[1, 0].set_title("motif 5")

axs[1, 1].plot(sel_shapelets.iloc[11, :], color="xkcd:sky blue")
axs[1, 1].set_title("shapelet 11")

axs[2, 0].plot(motif_df.iloc[2, :], color="blue")
axs[2, 0].set_title("motif 2")

axs[2, 1].plot(sel_shapelets.iloc[1, :], color="blue")
axs[2, 1].set_title("shapelet 1")

axs[3, 0].plot(motif_df.iloc[0, :], color="purple")
axs[3, 0].set_title("motif 0")

axs[3, 1].plot(sel_shapelets.iloc[1, :], color="purple")
axs[3, 1].set_title("shapelet 1")

fig.tight_layout()
plt.show()

sns.set()
fig, axs = plt.subplots(2, 2, figsize=(9, 3.5))
axs[0, 0].plot(motif_df.iloc[8, :], color="m")
axs[0, 0].set_title("motif 8")

axs[0, 1].plot(sel_shapelets.iloc[17, :], color="m")
axs[0, 1].set_title("shapelet 17")

axs[1, 0].plot(motif_df.iloc[0, :], color="pink")
axs[1, 0].set_title("motif 0")

axs[1, 1].plot(sel_shapelets.iloc[15, :], color="pink")
axs[1, 1].set_title("shapelet 15")

fig.tight_layout()
plt.show()

"""Stampo gli shaplet non ancora visualizzati """

sns.set()
fig, axs = plt.subplots(5, 2, figsize=(8, 7))
axs[0, 0].plot(sel_shapelets.iloc[2, :], color="blue")
axs[0, 0].set_title("shapelet 2")

axs[0, 1].plot(sel_shapelets.iloc[3, :], color="orange")
axs[0, 1].set_title("shapelet 3")

axs[1, 0].plot(sel_shapelets.iloc[4, :], color="green")
axs[1, 0].set_title("shapelet 4")

axs[1, 1].plot(sel_shapelets.iloc[6, :], color="red")
axs[1, 1].set_title("shapelet 6")

axs[2, 0].plot(sel_shapelets.iloc[7, :], color="purple")
axs[2, 0].set_title("shapelet 7")

axs[2, 1].plot(sel_shapelets.iloc[8, :], color="brown")
axs[2, 1].set_title("shapelet 8")

axs[3, 0].plot(sel_shapelets.iloc[9, :], color="pink")
axs[3, 0].set_title("shapelet 9")

axs[3, 1].plot(sel_shapelets.iloc[10, :], color="gray")
axs[3, 1].set_title("shapelet 23")

axs[4, 0].plot(sel_shapelets.iloc[12, :], color="m")
axs[4, 0].set_title("shapelet 12")

axs[4, 1].plot(sel_shapelets.iloc[13, :], color="xkcd:sky blue")
axs[4, 1].set_title("shapelet 13")
fig.tight_layout()
plt.show()

sns.set()
fig, axs = plt.subplots(4, 2, figsize=(8, 6))
axs[0, 0].plot(sel_shapelets.iloc[14, :], color="blue")
axs[0, 0].set_title("shapelet 14")

axs[0, 1].plot(sel_shapelets.iloc[16, :], color="orange")
axs[0, 1].set_title("shapelet 16")

axs[1, 0].plot(sel_shapelets.iloc[18, :], color="green")
axs[1, 0].set_title("shapelet 18")

axs[1, 1].plot(sel_shapelets.iloc[19, :], color="red")
axs[1, 1].set_title("shapelet 19")

axs[2, 0].plot(sel_shapelets.iloc[20, :], color="purple")
axs[2, 0].set_title("shapelet 20")

axs[2, 1].plot(sel_shapelets.iloc[21, :], color="brown")
axs[2, 1].set_title("shapelet 21")

axs[3, 0].plot(sel_shapelets.iloc[22, :], color="pink")
axs[3, 0].set_title("shapelet 22")

axs[3, 1].plot(sel_shapelets.iloc[23, :], color="gray")
axs[3, 1].set_title("shapelet 23")

fig.tight_layout()
plt.show()
