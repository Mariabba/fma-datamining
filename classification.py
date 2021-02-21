from pathlib import Path

import pandas as pd

import utils

genres = utils.load(Path("data/genres.csv"))
echonest = utils.load(Path("data/echonest.csv"))
tracks = utils.load(Path("data/tracks.csv"))
artists = utils.load(Path("data/raw_artists.csv"))

print(tracks.info())
print(tracks.head())
