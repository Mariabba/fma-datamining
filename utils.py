import ast
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import detect

try:
    from rich.progress import Progress, track
except ModuleNotFoundError:
    pass


def load(filepath, clean=False, dummies=False, fill=False):

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
    # check_rules(df, Path("data/rules.txt"))

    if clean:
        df = discretizer(df)
    if dummies:
        df = dummy_maker(df)
    if fill:
        df = fill_missing(df)

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

    # text analysis
    # album information ~ is used to state true as presence of information and false the absence
    df["album", "information"] = (~df["album", "information"].isnull()).astype(int)

    # artist bio ~ is used to state true as presence of bio and false the absence
    df["artist", "bio"] = (~df["artist", "bio"].isnull()).astype(int)

    # album producer ~ is used to state true as presence of producer and false the absence
    df["album", "producer"] = (~df["album", "producer"].isnull()).astype(int)

    # track comments
    bins = [-np.inf, 0, np.inf]
    labels = ["no_comments", "commented"]
    df["track", "comments"] = pd.cut(df["track", "comments"], bins=bins, labels=labels)

    # track duration
    bins = [-np.inf, 60, 120, np.inf]
    labels = ["1min", "2min", "+3min"]
    df["track", "duration"] = pd.cut(df["track", "duration"], bins=bins, labels=labels)

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

    # artist website - ~ is used to state true as presence of website stated and false the absence
    df["artist", "website"] = (~df["artist", "website"].isnull()).astype(int)

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

    # album engineer ~ is used to state true as presence of engineer and false the absence
    df["album", "engineer"] = (~df["album", "engineer"].isnull()).astype(int)

    # fill language_code
    df["track", "language_code"] = df["track", "language_code"].fillna(
        detect(str(df["track", "title"]))
    )

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


def fill_missing(df):
    pass

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
        print(rule)
        if rule == []:  # Useful for excessive line breaks at the end of file
            continue
        rest_of_rule = ""
        if len(rule) > 4:  # This checks if it's a complex rule
            rest_of_rule = f" {rule[4]} {rule[5]}"

        try:
            my_df = df.query(f"not (({rule[0]} {rule[1]}) {rule[2]} {rule[3]})").filter(
                [rule[0], rule[1]]
            )
            df_no_nan = my_df.dropna()

            errors = errors.append(
                {
                    "Rule": f"{rule[0]} {rule[1]} {rule[2]} {rule[3]}({rest_of_rule})",
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
                    "Rule": f"{rule[0]} {rule[1]} {rule[2]} {rule[3]}({rest_of_rule})",
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
