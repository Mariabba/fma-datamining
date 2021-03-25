from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as mso
from langdetect import detect

import utils

sns.set_theme(style="white")
tracks = utils.load("data/tracks.csv", dummies=True)

# fill dataCreated_ data released

# sistemare questione generi

y = mso.matrix(tracks)
print(y)
y = plt.savefig(Path("viz/missing_matrix.png"), bbox_inches="tight")

x = mso.bar(tracks)
print(x)
x = plt.savefig(Path("viz/missing_bar.png"), bbox_inches="tight")
