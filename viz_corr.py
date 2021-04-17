from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import utils

sns.set_theme(style="white")

tracks = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="discrete", fill=True, outliers=False
)
tracks.info()

# Encoding for the correlation
# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "comments"),
    ("album", "favorites"),
    ("album", "listens"),
    ("album", "type"),
    ("artist", "comments"),
    ("artist", "favorites"),
    ("track", "comments"),
    ("track", "favorites"),
    ("track", "license"),
    ("track", "listens"),
    ("track", "language_code"),
]
for col in column2encode:
    le = LabelEncoder()
    tracks[col] = le.fit_transform(tracks[col])
    label_encoders[col] = le
tracks.info()

# Compute the correlation matrix
corr = tracks.corr()
# print(corr["Attrition"])

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=1,
    center=0,
    vmin=-1,
    square=True,
    linewidths=1.1,
    cbar_kws={"shrink": 0.5},
)
# plt.show()
plt.title("Correlation Matrix")
plt.xlabel("Class name")
plt.ylabel("Class name")
plt.xticks(fontsize=10)
plt.savefig(Path("viz/correlation.png"), bbox_inches="tight")
