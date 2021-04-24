import pandas as pd
from matplotlib import pyplot as plt
from rich import pretty, print

import utils

pretty.install()
# genres = utils.load("data/genres.csv")
# echonest = utils.load("data/echonest.csv")
tracks = utils.load_tracks(
    buckets="continuous", dummies=True, fill=True, outliers=False
)
# artists = utils.load("data/raw_artists.csv")

print(tracks.info())


threshold = 0.9

low_coverage = []
for col in tracks:
    miao = tracks[col].isnull().mean()
    if miao > threshold:
        low_coverage.append(col)

print(low_coverage)
print(tracks[("track", "duration")].describe())

import seaborn as sns

fig = plt.subplots(figsize=(100, 20))
fig_dims = (1, 1)
ax = plt.subplot2grid(fig_dims, (0, 0))
sns.countplot(x=("track", "interest"), data=tracks, palette="hls")
plt.title("Frequency of duration")
plt.xticks(rotation=90)
plt.show()
# error checking
# errors = utils.check_rules(tracks, "data/rules.txt")
# print(errors)

# print("Here are informations on tracks")
# print(tracks.info())


# my_df = tracks.query(f"not ('album', 'listens') < 0")
# print(my_df)

# errors = utils.check_rules(tracks, Path("data/rules.txt"))
# print(errors)

# print(tracks["track", "comments"].describe())
# print(tracks["track", "comments"].head(30))

# print(tracks[("track", "dummy_lyricist")].value_counts())

# df_decisiontree = tracks[[blablabla]]
# df_knn = tracks[[blablaslsadasdldsaldsa]]

print(tracks[("album", "type")].unique())
