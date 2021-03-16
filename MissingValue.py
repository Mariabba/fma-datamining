from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as mso
import utils
from langdetect import detect

sns.set_theme(style="white")


tracks = utils.load(Path("data/tracks.csv"), clean=True, dummies=True)

# fill dataCreated_ data released

# sistemare questione generi

y = mso.matrix(tracks)
print(y)
y = plt.savefig(Path("viz/missing_matrix.png"), bbox_inches="tight")

x = mso.bar(tracks)
print(x)
x = plt.savefig(Path("viz/missing_bar.png"), bbox_inches="tight")
