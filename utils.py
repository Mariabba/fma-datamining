import ast

import pandas as pd
from rich.progress import Progress, track


def load(filepath, clean=False, dummies=False):

    filename = filepath.name

    if "features" in filename:
        # print("using features")
        df = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    elif "echonest" in filename:
        # print("using echonest")
        df = pd.read_csv(filepath, index_col=0, header=[0, 1])

    elif any(x in filename for x in ("genres", "albums", "artists")):
        # print("using anyof")
        df = pd.read_csv(filepath, index_col=0)

    elif "tracks" in filename:
        # print("using tracks")
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

    if clean:
        df = discretizer(df)
    if dummies:
        df = dummy_maker(df)

    return df


def discretizer(df):
    df["track", "listens"] = pd.qcut(
        df["track", "listens"], 4, labels=["low", "medium", "high", "superhigh"]
    )
    return df


def dummy_maker(df, threshold=0.9):
    """
    returns a new dataframe with dummy variables for columns with <10% coverage

    New dummies will have values 0 if original was NaN, 1 if it had a value
    """

    low_coverage = []
    for col in df:
        miao = df[col].isnull().mean()
        if miao > threshold:
            low_coverage.append(col)
    my_df = df[low_coverage]

    my_df = (~my_df.isna()).astype(int)

    my_df.columns = pd.MultiIndex.from_tuples(
        [(a, f"dummy_{b}") for a, b in my_df.columns]
    )

    return df.append(my_df, ignore_index=True)


def clean(df):
    pass
    # 1. discretizzare ciò che ha senso sia discretizzato

    # 2. provare a vedere se dà informazioni la dummizzazione delle colonne con coverage <10%

    # 3. matrice di correlazione

    # 4. check di consistenza: come?

    # 5. missing values colonne coverage > 80% : come riempire?
