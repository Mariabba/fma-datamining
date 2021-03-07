from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils

sns.set_theme(style="white")

tracks = utils.load(Path("data/tracks.csv"))

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
plt.savefig(Path("viz/correlation.png"), bbox_inches="tight")
