import pandas as pd
from rich import pretty, print

import utils

pretty.install()
# genres = utils.load("data/genres.csv")
# echonest = utils.load("data/echonest.csv")
tracks = utils.load(
    "data/tracks.csv", buckets="basic", dummies=False, fill=False, outliers=False
)
# artists = utils.load("data/raw_artists.csv")

print(tracks.info())
tracks = utils.load_tracks(buckets="discrete")
print(tracks.info())

"""
exit()
threshold = 0.9

low_coverage = []
for col in tracks:
    miao = tracks[col].isnull().mean()
    if miao > threshold:
        low_coverage.append(col)

print(low_coverage)

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
"""
