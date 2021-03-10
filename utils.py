import ast
import numpy as np
import pandas as pd

try:
    from rich.progress import Progress, track
except ModuleNotFoundError:
    pass


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
        df = correct_dtypes(df)

    else:
        print(f"Something bad just happened with {filename}.")

    df = df.convert_dtypes()

    # Deletion of columns
    del df[("set", "subset")]
    del df[("track", "bit_rate")]
    del df[("artist", "latitude")]
    del df[("artist", "longitude")]

    if clean:
        df = discretizer(df)
    if dummies:
        df = dummy_maker(df)

    return df


def correct_dtypes(df):
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

    return df


def discretizer(df):
    # album comments
    bins = [-np.inf, -1, 0, np.inf]
    labels = ["no_info", "no_comments", "commented"]
    df["album", "comments"] = pd.cut(df["album", "comments"], bins=bins, labels=labels)

    # artist comments
    bins = [-np.inf, -1, 0, np.inf]
    labels = ["no_info", "no_comments", "commented"]
    df["artist", "comments"] = pd.cut(
        df["artist", "comments"], bins=bins, labels=labels
    )

    print(df["artist", "comments"].value_counts())

    print(df["album", "favorites"].value_counts())

    # artist favorites
    bins = [-np.inf, -1, 0, 1, 2, 3, np.inf]
    labels = [
        "no_album",
        "no_favorites",
        "lowest_favorites",
        "low_favorites",
        "medium_favorites",
        "high_favorites",
    ]
    df["album", "favorites"] = pd.cut(
        df["album", "favorites"], bins=bins, labels=labels
    )

    print(df["album", "favorites"].value_counts())

    # text analysis
    # album information
    df["album", "information"] = ~df[
        "album", "information"
    ].isnull()  # ~ is used to state true as presence of information and false the absence
    print(df["album", "information"].value_counts())

    # artist bio
    df["artist", "bio"] = ~df[
        "artist", "bio"
    ].isnull()  # ~ is used to state true as presence of bio and false the absence
    print(df["artist", "bio"].value_counts())

    # album producer
    df["album", "producer"] = ~df[
        "album", "producer"
    ].isnull()  # ~ is used to state true as presence of producer and false the absence
    print(df["album", "producer"].value_counts())

    # track comments
    bins = [-np.inf, 0, np.inf]
    labels = ["no_comments", "commented"]
    df["track", "comments"] = pd.cut(df["track", "comments"], bins=bins, labels=labels)

    print(df["track", "comments"].value_counts())

    return df


def dummy_maker(df, threshold=0.9):
    """
    returns a new dataframe with dummy variables for columns with <10% coverage

    New dummies will have values 0 if original was NaN, 1 if it had a value
    Original columns are removed
    """

    low_coverage = []
    for col in df:
        miao = df[col].isnull().mean()
        if miao > threshold:
            low_coverage.append(col)
    my_df = df[low_coverage]
    df = df.drop(columns=low_coverage)

    my_df = (~my_df.isna()).astype(int)

    my_df.columns = pd.MultiIndex.from_tuples([(a, f"d_{b}") for a, b in my_df.columns])

    return pd.concat([df, my_df], axis=1)


def clean(df):
    pass
    # 1. discretizzare ciò che ha senso sia discretizzato

    # 2. provare a vedere se dà informazioni la dummizzazione delle colonne con coverage <10%

    # 3. matrice di correlazione

    # 4. check di consistenza: come?

    # 5. missing values colonne coverage > 80% : come riempire?
