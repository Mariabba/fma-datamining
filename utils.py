import pandas as pd
import os.path


def load(filepath):

    filename = os.path.basename(filepath)

    if "features" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "genres" or "albums" or "artists" in filename:
        return pd.read_csv(filepath, index_col=0)
