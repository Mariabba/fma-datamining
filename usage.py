from pathlib import Path

import pandas as pd

import utils

# genres = utils.load(Path("data/genres.csv"))
# echonest = utils.load(Path("data/echonest.csv"))
tracks = utils.load(Path("data/tracks.csv"))
# artists = utils.load(Path("data/raw_artists.csv"))
# print(tracks.info())

# tracks = utils.dummy_maker(tracks)
print(tracks.info())

tracks = utils.discretizer(tracks)
# print(tracks[("track", "dummy_lyricist")].value_counts())

del tracks[("set", "subset")]

# df_decisiontree = tracks[[blablabla]]
# df_knn = tracks[[blablaslsadasdldsaldsa]]
