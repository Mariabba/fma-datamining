import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from music import MusicDB

from tslearn.metrics import dtw

musi = MusicDB()
print(musi.df.info())

X = musi.df
y = musi.feat["enc_genre"]

print(musi.feat["genre"].unique())

sel_shapelets = pd.read_csv("musicshaplet.csv")

motif_df = pd.read_csv("musicmotif.csv")
motif_df = motif_df.drop(columns=["StartPoint", "MinMPDistance", "CentroidName"])

sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets.iloc[0, :], color="blue")
axs[0, 0].set_title("shaplet 0")

axs[0, 1].plot(sel_shapelets.iloc[1, :], color="orange")
axs[0, 1].set_title("shaplet 1")

axs[1, 0].plot(sel_shapelets.iloc[2, :], color="green")
axs[1, 0].set_title("shaplet 2")

axs[1, 1].plot(sel_shapelets.iloc[3, :], color="red")
axs[1, 1].set_title("shaplet 3")

axs[2, 0].plot(sel_shapelets.iloc[4, :], color="purple")
axs[2, 0].set_title("shaplet 4")

axs[2, 1].plot(sel_shapelets.iloc[5, :], color="brown")
axs[2, 1].set_title("shaplet 5")

axs[3, 0].plot(sel_shapelets.iloc[6, :], color="pink")
axs[3, 0].set_title("shaplet 6")

axs[3, 1].plot(sel_shapelets.iloc[7, :], color="gray")
axs[3, 1].set_title("shaplet 7")
fig.tight_layout()
plt.show()

sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets.iloc[8, :], color="blue")
axs[0, 0].set_title("shaplet 8")

axs[0, 1].plot(sel_shapelets.iloc[9, :], color="orange")
axs[0, 1].set_title("shaplet 9")

axs[1, 0].plot(sel_shapelets.iloc[10, :], color="green")
axs[1, 0].set_title("shaplet 10")

axs[1, 1].plot(sel_shapelets.iloc[11, :], color="red")
axs[1, 1].set_title("shaplet 11")

axs[2, 0].plot(sel_shapelets.iloc[12, :], color="purple")
axs[2, 0].set_title("shaplet 12")

axs[2, 1].plot(sel_shapelets.iloc[13, :], color="brown")
axs[2, 1].set_title("shaplet 13")

axs[3, 0].plot(sel_shapelets.iloc[14, :], color="pink")
axs[3, 0].set_title("shaplet 14")

axs[3, 1].plot(sel_shapelets.iloc[15, :], color="gray")
axs[3, 1].set_title("shaplet 15")
fig.tight_layout()
plt.show()

sns.set()
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs[0, 0].plot(sel_shapelets.iloc[16, :], color="blue")
axs[0, 0].set_title("shaplet 16")

axs[0, 1].plot(sel_shapelets.iloc[17, :], color="orange")
axs[0, 1].set_title("shaplet 17")

axs[1, 0].plot(sel_shapelets.iloc[18, :], color="green")
axs[1, 0].set_title("shaplet 18")

axs[1, 1].plot(sel_shapelets.iloc[19, :], color="red")
axs[1, 1].set_title("shaplet 19")

axs[2, 0].plot(sel_shapelets.iloc[20, :], color="purple")
axs[2, 0].set_title("shaplet 20")

axs[2, 1].plot(sel_shapelets.iloc[21, :], color="brown")
axs[2, 1].set_title("shaplet 21")

axs[3, 0].plot(sel_shapelets.iloc[22, :], color="pink")
axs[3, 0].set_title("shaplet 22")

axs[3, 1].plot(sel_shapelets.iloc[23, :], color="gray")
axs[3, 1].set_title("shaplet 23")
fig.tight_layout()
plt.show()

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

bestemmia = pd.DataFrame(data={"Motif/Shaplet": dio, "DTW": dist}, index=dio)
print(bestemmia.info())

bestemmia = bestemmia.sort_values(by="DTW")

print(bestemmia.head(10))

bestemmia["DTW"].head(10).plot(
        kind="bar"
    )
plt.xticks(rotation=20)
plt.show()
