import pandas as pd
import os.path


def load(filepath):

    filename = os.path.basename(filepath)

    if "features" in filename:
        print(filename)
        print("1")
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        print(filename)
        print("2")
        return pd.read_csv(filepath, index_col=0, header=[0, 1])

    if ("genres" or "albums" or "artists") in filename:
        print(filename)
        print("3")
        return pd.read_csv(filepath, index_col=0)

    if "tracks" in filename:
        print(filename)
        print("4")
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1], low_memory=False)
        return tracks
