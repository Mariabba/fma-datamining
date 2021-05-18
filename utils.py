import ast
from pathlib import Path

import numpy as np
import pandas as pd

# from langdetect import detect
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from rich import print
except ModuleNotFoundError:
    pass


def load_tracks_xyz(
    filepath="data/tracks.csv",
    splits=3,
    buckets="basic",
    dummies=True,
    fill=True,
    outliers=True,
    extractclass=None,
    small=False,
    _good=True,
) -> dict:
    docstring = """
    Same usage as load_tracks(), except for these differences:
    Returns dict of pd.Dataframe from tracks.csv: ["train", "vali", "test"]

    If splits = 2, returns dict of pd.Dataframe from tracks.csv: ["train", "test"]

    If extractclass=column : returns a dict of [train_x, train_y, vali_x, vali_y, test_x, test_y]
    or if extractclass=column, splits = 2 : returns dict of [train_x, train_y, test_x, test_y]
    """
    if small:
        df = load_small_tracks(
            filepath, buckets, dummies, fill, outliers, _good, _xyz=True
        )
    else:
        df = load(filepath, buckets, dummies, fill, outliers, _good)

    # split train, vali, test
    mask_train = df[("set", "split")] == "training"
    df_train = df[mask_train]
    del df_train[("set", "split")]

    if splits == 3:
        mask_vali = df[("set", "split")] == "validation"
        mask_test = df[("set", "split")] == "test"
        df_vali = df[mask_vali]
        df_test = df[mask_test]
        del df_vali[("set", "split")]
        del df_test[("set", "split")]
        all_dfs = {"train": df_train, "vali": df_vali, "test": df_test}
    elif splits == 2:
        mask_test = (df[("set", "split")] == "test") | (
            df[("set", "split")] == "validation"
        )
        df_test = df[mask_test]
        del df_test[("set", "split")]
        all_dfs = {"train": df_train, "test": df_test}
    else:
        raise ValueError(docstring)

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


def load_small_tracks(
    filepath="data/tracks.csv",
    buckets="basic",
    dummies=True,
    fill=True,
    outliers=True,
    _good=True,
    _xyz=False,
) -> pd.DataFrame:
    df = load(filepath, buckets, dummies, fill, outliers, _good)

    columns_to_keep = [
        ("album", "type"),
        ("artist", "website"),
        ("album", "producer"),
        ("artist", "bio"),
        ("album", "information"),
        ("album", "engineer"),
        ("artist", "active_year_end"),
        ("track", "publisher"),
        ("track", "duration"),  # continuous from here on out
        ("track", "listens"),
        ("track", "interest"),
        ("set", "split"),
    ]

    df = df[columns_to_keep]

    if not _xyz:
        del df[("set", "split")]
    return df


def load_tracks(
    filepath="data/tracks.csv",
    buckets="basic",
    dummies=True,
    fill=True,
    outliers=True,
    _good=True,
    givegenre=False,
) -> pd.DataFrame:
    docstring = """
    usage: load_tracks(string filepath default data/tracks.csv,
    buckets='basic|continuous|discrete' default basic,
    dummies=True|False default True,
    fill=True|False default True,
    outliers=True|False default True
    """

    df = load(filepath, buckets, dummies, fill, outliers, _good, givegenre)
    del df[("set", "split")]

    return df


def load(
    filepath: str,
    buckets="basic",
    dummies=False,
    fill=False,
    outliers=False,
    _good=False,
    givegenre=False,
) -> pd.DataFrame:
    """
    private function
    """

    # integrity check
    if buckets not in ["basic", "continuous", "discrete"]:
        raise ValueError(
            "You passed a wrong 'buckets' parameter. Please refer to README.md for documentation."
        )

    if not _good:
        print(
            """\n[magenta]Warning: utils.load() should be a private method. :vampire:
         Use utils.load_tracks() or utils.load_tracks_xyz() instead.
         See README.md for usage.
         This will still work as usual anyways. :kiss: [/magenta]\n"""
        )

    pipi = check_pickle(buckets, dummies, fill, outliers, givegenre)
    if pipi["status"]:  # must be boolean
        return pipi["df"]  # must be dataframe

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
        ("album", "tags"),  # meh
        ("album", "title"),
        ("album", "tracks"),  # meh
        ("artist", "active_year_begin"),
        ("artist", "associated_labels"),  # meh
        ("artist", "id"),
        ("artist", "latitude"),
        ("artist", "location"),  # meh
        ("artist", "longitude"),
        ("artist", "members"),  # meh
        ("artist", "name"),
        ("artist", "related_projects"),  # meh
        ("artist", "tags"),  # meh
        ("set", "subset"),
        ("track", "bit_rate"),  # meh
        ("track", "date_recorded"),
        ("track", "genres"),
        ("track", "number"),  # meh
        ("track", "tags"),  # meh
    ]
    df = df.drop(columns2drop, axis=1)

    # Rows we're not interested in for any method
    df = df[df["album", "type"] != "Contest"]

    # start of parameter choices
    if buckets == "basic":
        pass
    elif buckets == "continuous":
        df = discretizer_continuousmethods(df)
    elif buckets == "discrete":
        df = discretizer_discretemethods(df)
    else:
        raise ValueError
    df = discretizer(df)

    if dummies:
        df = dummy_maker(df)
    if fill:
        df = fill_missing(df)
    if outliers:
        df = treat_outliers(df)

    # delete columns that we used in some method
    del df[("track", "title")]
    del df[("track", "genres_all")]
    if not givegenre:
        del df[("track", "genre_top")]

    df.attrs["df_name"] = filename
    save_pickle(df, buckets, dummies, fill, outliers, givegenre)
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
        0: "top genre missing",
    }

    def search_for_life(value) -> str:
        for element in value:
            if element in toplvl_genres:
                return toplvl_genres[element]
        return toplvl_genres[0]

    def search_for_death(value) -> list:
        results = []
        for element in value:
            if element in toplvl_genres:
                results.append(toplvl_genres[element])
        if results:
            return results
        else:
            return [toplvl_genres[0]]

    s = df[("track", "genres_all")]
    s = s.apply(search_for_death)

    mlb = MultiLabelBinarizer()
    dumm = mlb.fit_transform(s)
    index_columns = pd.MultiIndex.from_product([["genre"], mlb.classes_])

    my_dummies = pd.DataFrame(dumm, columns=index_columns, index=df.index)

    df = pd.concat([df, my_dummies], axis=1)

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

    # elimino le row che hanno valori mancanti
    df = df[df[("album", "type")].notna()]
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
        df.shape[0] == 99391
    ), "treat_outliers only tested with dummies=True, fill=True - did we recently change the shape of df?"

    df_outliers = pd.read_csv("strange_results/abod1072.csv")
    df_outliers = df_outliers.set_index(df.index)

    return df[df_outliers["0"] == 0]


def check_pickle(buckets, dummies, fill, outliers, givegenre):
    pipi = {"status": False, "df": None}
    path_to_pickle = make_pick_encoding(buckets, dummies, fill, outliers, givegenre)

    try:
        pipi["df"] = pd.read_pickle(path_to_pickle)
        pipi["status"] = True
    except FileNotFoundError:
        pass

    return pipi


def save_pickle(df, buckets, dummies, fill, outliers, givegenre):
    path_to_pickle = make_pick_encoding(buckets, dummies, fill, outliers, givegenre)
    df.to_pickle(path_to_pickle)


def make_pick_encoding(buckets, dummies, fill, outliers, givegenre) -> Path:
    """
    Remember that I put tracks hardcoded in here. If necessary other csv than tracks, must change it with
    filename logic like in load()
    """
    encoded = "-"
    encoded += "0" if buckets == "basic" else "1" if buckets == "continuous" else "2"
    encoded += "1" if dummies else "0"
    encoded += "1" if fill else "0"
    encoded += "1" if outliers else "0"
    encoded += "1" if givegenre else "0"
    encoded = "data/picks/tracks" + encoded + ".pkl"
    path_to_pickle = Path(encoded)

    return path_to_pickle
