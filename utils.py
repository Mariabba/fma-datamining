import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import detect

try:
    from rich.progress import Progress, track
except ModuleNotFoundError:
    pass


def load(filepath, buckets="basic", dummies=False, fill=False, buckets_knn=False):
    """
    usage: load(Path object filepath,
    buckets='basic|continuous|discrete' default basic,
    dummies=True|False default False,
    fill=True|False default False
    """
    filename = Path(filepath).name

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

    # start of parameter choices
    if buckets == "basic":
        df = discretizer(df)
    elif buckets == "continuous":
        df = discretizer(df)
        df = discretizer_continuousmethods(df)
    elif buckets == "discrete":
        df = discretizer(df)
        df = discretizer_discretemethods(df)
    else:
        raise ValueError(
            "usage: load(Path object filepath, buckets='basic|continuous|discrete' default basic, dummies=True|False default False, fill=True|False default False"
        )

    if dummies:
        df = dummy_maker(df)
    if fill:
        df = fill_missing(df)
    if buckets_knn:
        df = discretizer_knn(df)

    return df


def correct_dtypes(df):
    """
    Used for tracks.csv to convert into category, datetime types
    """
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
    """
    General discretizations that we use for EVERYTHING.
    """
    # datetime manipulation
    df[("album", "date_created")] = pd.to_datetime(
        df[("album", "date_created")]
    ).dt.year
    df[("artist", "date_created")] = pd.to_datetime(
        df[("artist", "date_created")]
    ).dt.year
    df[("track", "date_created")] = pd.to_datetime(
        df[("track", "date_created")]
    ).dt.year

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

    # album favorites
    bins = [-np.inf, -1, 0, 2, 5, np.inf]
    labels = [
        "no_info",
        "no_favorites",
        "low_favorites",
        "medium_favorites",
        "high_favorites",
    ]
    df["album", "favorites"] = pd.cut(
        df["album", "favorites"], bins=bins, labels=labels
    )

    # artist favorites
    bins = [-np.inf, 0, 10, 50, 150, 500, np.inf]
    labels = [
        "no_favorites",
        "low_favorites",
        "medium_favorites",
        "high_favorites",
        "higher_favorites",
        "super_favourites",
    ]
    df["artist", "favorites"] = pd.cut(
        df["artist", "favorites"], bins=bins, labels=labels
    )

    # track comments
    bins = [-np.inf, 0, np.inf]
    labels = ["no_comments", "commented"]
    df["track", "comments"] = pd.cut(df["track", "comments"], bins=bins, labels=labels)

    # removed track duration discretization better performance decision tree
    """
    # track duration
    bins = [-np.inf, 60, 120, np.inf]
    labels = ["1min", "2min", "+3min"]
    df["track", "duration"] = pd.cut(df["track", "duration"], bins=bins, labels=labels)
    """

    # track favorites
    bins = [-np.inf, 0, 2, 5, np.inf]
    labels = [
        "no_favorites",
        "low_favorites",
        "medium_favorites",
        "high_favorites",
    ]
    df["track", "favorites"] = pd.cut(
        df["track", "favorites"], bins=bins, labels=labels
    )

    # album listens
    bins = [-np.inf, -1, 10000, 50000, 150000, np.inf]
    labels = [
        "no_info",
        "low_listened",
        "medium_listened",
        "high_listened",
        "higher_listened",
    ]
    df["album", "listens"] = pd.cut(df["album", "listens"], bins=bins, labels=labels)

    # track listens
    bins = [-np.inf, 1000, 5000, np.inf]
    labels = [
        "low_listened",
        "medium_listened",
        "high_listened",
    ]
    df["track", "listens"] = pd.cut(df["track", "listens"], bins=bins, labels=labels)

    # fill language_code
    df["track", "language_code"] = df["track", "language_code"].fillna(
        detect(str(df["track", "title"]))
    )

    return df


def discretizer_continuousmethods():
    """
    Discretizations that we use for methods that accept or require continuous variables (like knn)
    """
    pass


def discretizer_discretemethods():
    """
    Discretizations that we use for methods that accept or require discrete variables (like xxx)
    """
    pass


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

    # Special cases
    df["album", "engineer"] = (~df["album", "ds_engineer"].isnull()).astype(int)
    df["album", "information"] = (~df["album", "ds_information"].isnull()).astype(int)
    df["artist", "bio"] = (~df["artist", "ds_bio"].isnull()).astype(int)
    df["album", "producer"] = (~df["album", "ds_producer"].isnull()).astype(int)
    df["artist", "website"] = (~df["artist", "ds_website"].isnull()).astype(int)

    return pd.concat([df, my_df], axis=1)


def fill_missing(df):
    # Deletion of columns
    del df[("set", "subset")]
    del df[("track", "bit_rate")]
    del df[("artist", "latitude")]
    del df[("artist", "longitude")]
    del df[("artist", "active_year_begin")]
    del df[("artist", "associated_labels")]
    del df[("artist", "related_projects")]

    # eliminate per andare avanti che potrebbero servire in seguito
    del df[("album", "date_released")]
    del df[("artist", "location")]
    del df[("artist", "members")]
    del df[("track", "genre_top")]

    # elimino le row che non hanno album type
    df = df[df[("album", "type")].notna()]

    # elimino le row che hanno valori mancanti in artist_date_created , track:license e track title
    df = df[df[("artist", "date_created")].notna()]
    df = df[df[("track", "license")].notna()]
    df = df[df[("track", "title")].notna()]

    return df


def check_rules(df: pd.DataFrame, rules_path: Path) -> pd.DataFrame:
    with open(rules_path, "r") as reader:
        file_contents = reader.readlines()
    rules = [x.split() for x in file_contents]

    errors = pd.DataFrame(
        columns=[
            "Rule",
            "Errors",
            "Errors %",
            "Errors without nan",
            "without nan %",
            "Records",
        ]
    )
    for rule in rules:
        if rule == []:  # Useful for excessive line breaks
            continue

        try:
            my_df = df[df.loc[:, (rule[0], rule[1])] < 0]

            df_no_nan = my_df.dropna()

            errors = errors.append(
                {
                    "Rule": f"{rule[0]} {rule[1]} {rule[2]} {rule[3]}",
                    "Errors": len(my_df),
                    "Errors %": len(my_df) / len(df) * 100,
                    "Errors without nan": len(df_no_nan),
                    "without nan %": len(df_no_nan) / len(df) * 100,
                    "Records": [x for x in my_df.index],
                },
                ignore_index=True,
            )
        except pd.core.computation.ops.UndefinedVariableError:
            errors = errors.append(
                {
                    "Rule": f"{rule[0]} {rule[1]} {rule[2]} {rule[3]}",
                    "Errors": None,
                    "Errors %": None,
                    "Errors without nan": None,
                    "without nan %": None,
                    "Records": None,
                },
                ignore_index=True,
            )
    return errors


def clean(df):
    pass

    # 4. check di consistenza: come?

    # 5. missing values colonne coverage > 80% : come riempire?


def discretizer_knn(df):
    # album information ~ is used to state true as presence of information and false the absence
    df["album", "information"] = (~df["album", "information"].isnull()).astype(int)

    # artist bio ~ is used to state true as presence of bio and false the absence
    df["artist", "bio"] = (~df["artist", "bio"].isnull()).astype(int)

    # album producer ~ is used to state true as presence of producer and false the absence
    df["album", "producer"] = (~df["album", "producer"].isnull()).astype(int)

    # artist website - ~ is used to state true as presence of website stated and false the absence
    df["artist", "website"] = (~df["artist", "website"].isnull()).astype(int)

    # album listens #TODO rivedere se tenerla così o eliminare discretizzazione
    bins = [-np.inf, -1, 10000, 50000, 150000, np.inf]
    labels = [
        "no_info",
        "low_listened",
        "medium_listened",
        "high_listened",
        "higher_listened",
    ]
    df["album", "listens"] = pd.cut(df["album", "listens"], bins=bins, labels=labels)

    # album engineer ~ is used to state true as presence of engineer and false the absence
    df["album", "engineer"] = (~df["album", "engineer"].isnull()).astype(int)

    # fill language_code
    df["track", "language_code"] = df["track", "language_code"].fillna(
        detect(str(df["track", "title"]))
    )

    return df
