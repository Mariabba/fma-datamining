import pandas as pd
import utils

# tracks = utils.load("data/tracks.csv")
# genres = utils.load("data/genres.csv")
# features = utils.load("data/features.csv")
artists = utils.load("data/raw_artists.csv")
genres = utils.load("data/genres.csv")
echonest = utils.load("data/echonest.csv")
tracks = utils.load("data/tracks.csv")

print(echonest.info())
print(echonest.head())
