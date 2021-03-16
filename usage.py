from pathlib import Path

import pandas as pd
from rich import pretty, print

import utils

pretty.install()
# genres = utils.load(Path("data/genres.csv"))
# echonest = utils.load(Path("data/echonest.csv"))
tracks = utils.load(Path("data/tracks.csv"), clean=True, dummies=True)
# artists = utils.load(Path("data/raw_artists.csv"))


print("Here are informations on tracks")
print(tracks.info())
print("ciao")

# my_df = tracks.query(f"not ('album', 'listens') < 0")
# print(my_df)

# errors = utils.check_rules(tracks, Path("data/rules.txt"))
# print(errors)

# print(tracks["track", "comments"].describe())
# print(tracks["track", "comments"].head(30))

# print(tracks[("track", "dummy_lyricist")].value_counts())

# df_decisiontree = tracks[[blablabla]]
# df_knn = tracks[[blablaslsadasdldsaldsa]]
