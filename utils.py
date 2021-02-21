import ast

import pandas as pd


def load(filepath):

    filename = filepath.name

    if "features" in filename:
        print("using features")
        df = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    elif "echonest" in filename:
        print("using echonest")
        df = pd.read_csv(filepath, index_col=0, header=[0, 1])

    elif any(x in filename for x in ("genres", "albums", "artists")):
        print("using anyof")
        df = pd.read_csv(filepath, index_col=0)

    elif "tracks" in filename:
        print("using tracks")
        df = pd.read_csv(filepath, index_col=0, header=[0, 1], low_memory=False)

        columns = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in columns:
            df[column] = df[column].map(ast.literal_eval)

        columns = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in columns:
            df[column] = pd.to_datetime(df[column])

        columns = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in columns:
            df[column] = df[column].astype("category")
    else:
        print(f"Something bad just happened with {filename}.")

    df = df.convert_dtypes()
    return df
