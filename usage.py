from pathlib import Path

import pandas as pd

import utils

# genres = utils.load(Path("data/genres.csv"))
# echonest = utils.load(Path("data/echonest.csv"))
tracks = utils.load(Path("data/tracks.csv"))
# artists = utils.load(Path("data/raw_artists.csv"))

tracks = utils.dummy_maker(tracks)

tracks = utils.discretizer(tracks)

print("Here are informations on tracks")
print(tracks.info())
# print(tracks["track", "comments"].describe())
# print(tracks["track", "comments"].head(30))

# print(tracks[("track", "dummy_lyricist")].value_counts())

# df_decisiontree = tracks[[blablabla]]
# df_knn = tracks[[blablaslsadasdldsaldsa]]
