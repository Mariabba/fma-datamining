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


def load_tracks_xyz(
    filepath="data/tracks.csv",
    buckets="basic",
    dummies=True,
    fill=True,
    outliers=True,
    extractclass=None,
) -> dict:
    """
    Same usage as load(), with everything turned on though
    Returns tuple of pd.Dataframe from tracks.csv: (train_df, validation_df, test_df)

    If extractclass=column : returns a dict of [train_x, train_y, vali_x, vali_y, test_x, test_y]
    """
    df = load(filepath, buckets, dummies, fill, outliers)

    # split train, vali, test
    mask_train = df[("set", "split")] == "training"
    mask_vali = df[("set", "split")] == "validation"
    mask_test = df[("set", "split")] == "test"
    df_train = df[mask_train]
    df_vali = df[mask_vali]
    df_test = df[mask_test]
    del df_train[("set", "split")]
    del df_vali[("set", "split")]
    del df_test[("set", "split")]

    all_dfs = {"train": df_train, "vali": df_vali, "test": df_test}

    # extractclass
    if not extractclass:
        return all_dfs
    else:
        results = {}
        for key, dataf in all_dfs.items():
            attributes = [col for col in dataf.columns if col != extractclass]
            results[f"{key}_x"] = dataf[attributes]
            results[f"{key}_y"] = dataf[extractclass]
        return results


def load_tracks(
    filepath="data/tracks.csv",
    buckets="basic",
    dummies=False,
    fill=False,
    outliers=False,
) -> pd.DataFrame:
    return load(filepath, buckets, dummies, fill, outliers)


def load(
    filepath: str, buckets="basic", dummies=False, fill=False, outliers=False
) -> pd.DataFrame:
    docstring = """
    usage: load(string filepath,
    buckets='basic|continuous|discrete' default basic,
    dummies=True|False default False,
    fill=True|False default False
    """
    filepath = Path(filepath)
    filename = filepath.name

    if "features" in filename:
        df = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
        df.name = filename

    elif "echonest" in filename:
        df = pd.read_csv(filepath, index_col=0, header=[0, 1])

    elif any(x in filename for x in ("genres", "albums", "artists")):
        df = pd.read_csv(filepath, index_col=0)

    elif "tracks" in filename:
        df = pd.read_csv(filepath, index_col=0, header=[0, 1], low_memory=False)
        df = correct_dtypes(df)

    else:
        raise ValueError(f"{filename} is not supported by load().")

    df = df.convert_dtypes()

    # Columns that we're not interested in for ANY method
    columns2drop = [
        ("album", "date_released"),  # meh
        ("album", "id"),
        ("album", "title"),
        ("album", "tracks"),
        ("artist", "active_year_begin"),
        ("artist", "id"),
        ("artist", "latitude"),
        ("artist", "longitude"),
        ("artist", "members"),  # meh
        ("artist", "name"),
        ("set", "subset"),
        ("track", "bit_rate"),  # meh
        ("track", "date_recorded"),
    ]
    df = df.drop(columns2drop, axis=1)

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
        raise ValueError(docstring)

    if dummies:
        df = dummy_maker(df)
    if fill:
        df = fill_missing(df)
    if outliers:
        df = treat_outliers(df)

    df = df[df["album", "type"] != "Contest"]
    df.attrs["df_name"] = filename
    return df


def correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
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


def discretizer(df: pd.DataFrame) -> pd.DataFrame:
    """
    General discretizations that we use for EVERYTHING.
    """
    # datetime manipulation
    df[("album", "date_created")] = pd.to_datetime(
        df[("album", "date_created")]
    ).dt.year.astype("Int64")
    df[("artist", "date_created")] = pd.to_datetime(
        df[("artist", "date_created")]
    ).dt.year.astype("Int64")
    df[("track", "date_created")] = pd.to_datetime(
        df[("track", "date_created")]
    ).dt.year.astype("Int64")

    # here we treat genres
    toplvl_genres = {
        12: "Rock",
        15: "Electronic",
        38: "Experimental",
        21: "Hip-Hop",
        17: "Folk",
        1235: "Instrumental",
        10: "Pop",
        2: "International",
        5: "Classical",
        8: "Old-Time / Historic",
        4: "Jazz",
        9: "Country",
        14: "Soul-RnB",
        20: "Spoken",
        3: "Blues",
        13: "Easy Listening",
    }

    miao = df[[("track", "genres_all")]]
    miao.columns = miao.columns.get_level_values(1)

    sc = set(["[", "]", "'", "'", " "])
    for index, row in miao.iterrows():
        temp = "".join([c for c in row["genres_all"] if c not in sc])
        row["genres_all"] = temp.split(",")

    for index, row in miao.iterrows():
        for element in toplvl_genres:
            if element in row["genres_all"]:
                row["only_top"] = element

    # miao = miao.eval("only_top=x if x in genres_all")
    # print(miao)
    return df


def discretizer_continuousmethods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretizations that we use for methods that prefer or require continuous variables (like KNN)
    """
    pass

    return df


def discretizer_discretemethods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretizations that we use for methods that prefer or require discrete variables (like xxx)
    """
    bins = [-np.inf, -1, 0, np.inf]
    labels = ["no_info", "no_comments", "commented"]
    df["album", "comments"] = pd.cut(df["album", "comments"], bins=bins, labels=labels)

    bins = [-np.inf, -1, 0, np.inf]
    labels = ["no_info", "no_comments", "commented"]
    df["artist", "comments"] = pd.cut(
        df["artist", "comments"], bins=bins, labels=labels
    )

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

    bins = [-np.inf, -1, 10000, 50000, 150000, np.inf]
    labels = [
        "no_info",
        "low_listened",
        "medium_listened",
        "high_listened",
        "higher_listened",
    ]
    df["album", "listens"] = pd.cut(df["album", "listens"], bins=bins, labels=labels)

    bins = [-np.inf, 1000, 5000, np.inf]
    labels = [
        "low_listened",
        "medium_listened",
        "high_listened",
    ]
    df["track", "listens"] = pd.cut(df["track", "listens"], bins=bins, labels=labels)

    return df


def dummy_maker(df, threshold=0.9) -> pd.DataFrame:
    """
    returns a new dataframe with dummy variables for columns with <10% coverage

    New dummies will have values 0 if original was NaN, 1 if it had a value
    Original columns are removed
    """

    # Columns coverage < threshold
    low_coverage = []
    for col in df:
        miao = df[col].isnull().mean()
        if miao > threshold:
            low_coverage.append(col)

    # Special cases
    special_cases = [
        ("album", "engineer"),
        ("album", "information"),
        ("artist", "bio"),
        ("album", "producer"),
        ("artist", "website"),
        # ("album", "type"),
    ]

    low_coverage.extend(special_cases)
    my_df = df[low_coverage]
    df = df.drop(columns=low_coverage)

    my_df = (~my_df.isna()).astype(int)
    my_df.columns = pd.MultiIndex.from_tuples(
        [(a, f"{b}") for a, b in my_df.columns]
    )  # was (a, f"d_{b}")

    return pd.concat([df, my_df], axis=1)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    # fill language_code
    df["track", "language_code"] = df["track", "language_code"].fillna(
        detect(str(df["track", "title"]))
    )

    # Deletion of columns
    del df[("artist", "associated_labels")]
    del df[("artist", "related_projects")]

    # eliminate per andare avanti che potrebbero servire in seguito
    del df[("artist", "location")]
    del df[("track", "genre_top")]

    # elimino le row che non hanno album type
    df = df[df[("album", "type")].notna()]

    # elimino le row che hanno valori mancanti in artist_date_created , track:license e track title
    df = df[df[("artist", "date_created")].notna()]
    df = df[df[("track", "license")].notna()]
    df = df[df[("album", "date_created")].notna()]

    return df


def check_rules(df: pd.DataFrame, rules_path: str) -> pd.DataFrame:
    """
    Checks rules_path (possibly 'data/rules.txt') and returns a dataframe with the error count for each rule
    """
    with open(Path(rules_path), "r") as reader:
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


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inserire qui trattamento outlier. Accetta dataframe e deve restituire dataframe.
    Testato solo con: buckets="basic", dummies=True, fill=True
    """
    assert (
        df.shape[0] == 99404
    ), "treat_outliers only tested with dummies=True, fill=True"

    df_outliers = pd.read_csv("strange_results/abod1072.csv")
    df_outliers = df_outliers.set_index(df.index)

    return df[df_outliers["0"] == 0]
