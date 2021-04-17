import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    plot_confusion_matrix,
)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from collections import defaultdict
import utils
import wittgenstein as lw

from sklearn.model_selection import train_test_split


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Rules Based Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Reds", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


class_name = ("album", "type")

df = utils.load_small_tracks(buckets="discrete")

attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep="=")
dfY = df[class_name]
df = pd.concat([dfX, dfY], axis=1)
print(df.info())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

df_prediction_train = pd.DataFrame()
df_prediction_test = pd.DataFrame()
df_prediction_train["y_train"] = y_train
df_prediction_test["y_test"] = y_test


def SingleTracks(
    X_train, X_test, y_train, y_test, df_prediction_train, df_prediction_test
):
    print("Prediction SingleTracks:")
    y_train = y_train.replace(
        ["Album", "Single Tracks", "Live Performance", "Radio Program"],
        [1, 0, 1, 1],
    )
    y_test = y_test.replace(
        ["Album", "Single Tracks", "Live Performance", "Radio Program"],
        [1, 0, 1, 1],
    )
    ripper_clf_ST = lw.RIPPER(k=1, prune_size=0.33)
    ripper_clf_ST.fit(X_train, y_train, class_feat=("album", "type"), pos_class=0)
    print(ripper_clf_ST)

    # Collect performance metrics
    precision = ripper_clf_ST.score(X_test, y_test, precision_score)
    recall = ripper_clf_ST.score(X_test, y_test, recall_score)
    cond_count = ripper_clf_ST.ruleset_.count_conds()
    print(ripper_clf_ST.ruleset_.out_pretty())
    print(ripper_clf_ST.ruleset_)
    print(f"precision: {precision} recall: {recall} conds: {cond_count}")
    y_pred_train = ripper_clf_ST.predict(X_train)
    y_pred_test = ripper_clf_ST.predict(X_test)

    df_prediction_train["SingleTracks_y_pred_train"] = y_pred_train
    df_prediction_test["SingleTracks_y_pred_test"] = y_pred_test
    return df_prediction_train, df_prediction_test


def LivePerformance(
    X_train, X_test, y_train, y_test, df_prediction_train, df_prediction_test
):
    print("Prediction LivePerformance:")
    y_train = y_train.replace(
        ["Album", "Single Tracks", "Live Performance", "Radio Program"],
        [1, 1, 0, 1],
    )
    y_test = y_test.replace(
        ["Album", "Single Tracks", "Live Performance", "Radio Program"],
        [1, 1, 0, 1],
    )
    ripper_clf_LP = lw.RIPPER(k=1, prune_size=0.33)
    ripper_clf_LP.fit(X_train, y_train, class_feat=("album", "type"), pos_class=0)
    print(ripper_clf_LP)

    # Collect performance metrics
    precision = ripper_clf_LP.score(X_test, y_test, precision_score)
    recall = ripper_clf_LP.score(X_test, y_test, recall_score)
    cond_count = ripper_clf_LP.ruleset_.count_conds()
    print(ripper_clf_LP.ruleset_.out_pretty())
    print(ripper_clf_LP.ruleset_)
    print(f"precision: {precision} recall: {recall} conds: {cond_count}")
    y_pred_train = ripper_clf_LP.predict(X_train)
    y_pred_test = ripper_clf_LP.predict(X_test)

    df_prediction_train["LivePerformance_y_pred_train"] = y_pred_train
    df_prediction_test["LivePerformance_y_pred_test"] = y_pred_test
    return df_prediction_train, df_prediction_test


def RadioProgram(
    X_train, X_test, y_train, y_test, df_prediction_train, df_prediction_test
):
    print("Prediction RadioProgram:")
    y_train = y_train.replace(
        ["Album", "Single Tracks", "Live Performance", "Radio Program"],
        [1, 1, 1, 0],
    )
    y_test = y_test.replace(
        ["Album", "Single Tracks", "Live Performance", "Radio Program"],
        [1, 1, 1, 0],
    )
    ripper_clf_RP = lw.RIPPER(k=1, prune_size=0.33)
    ripper_clf_RP.fit(X_train, y_train, class_feat=("album", "type"), pos_class=0)
    print(ripper_clf_RP)

    # Collect performance metrics
    precision = ripper_clf_RP.score(X_test, y_test, precision_score)
    recall = ripper_clf_RP.score(X_test, y_test, recall_score)
    cond_count = ripper_clf_RP.ruleset_.count_conds()
    print(ripper_clf_RP.ruleset_.out_pretty())
    print(ripper_clf_RP.ruleset_)
    print(f"precision: {precision} recall: {recall} conds: {cond_count}")
    y_pred_train = ripper_clf_RP.predict(X_train)
    y_pred_test = ripper_clf_RP.predict(X_test)

    df_prediction_train["RadioProgram_y_pred_train"] = y_pred_train
    df_prediction_test["RadioProgram_y_pred_test"] = y_pred_test
    return df_prediction_train, df_prediction_test


def Album(X_train, X_test, y_train, y_test, df_prediction_train, df_prediction_test):
    print("Prediction Album:")
    y_train = y_train.replace(
        ["Album", "Single Tracks", "Live Performance", "Radio Program"],
        [0, 1, 1, 1],
    )
    y_test = y_test.replace(
        ["Album", "Single Tracks", "Live Performance", "Radio Program"],
        [0, 1, 1, 1],
    )
    ripper_clf_A = lw.RIPPER(k=1, prune_size=0.33)
    ripper_clf_A.fit(X_train, y_train, class_feat=("album", "type"), pos_class=0)
    print(ripper_clf_A)

    # Collect performance metrics
    precision = ripper_clf_A.score(X_test, y_test, precision_score)
    recall = ripper_clf_A.score(X_test, y_test, recall_score)
    cond_count = ripper_clf_A.ruleset_.count_conds()
    print(ripper_clf_A.ruleset_.out_pretty())
    print(ripper_clf_A.ruleset_)
    print(f"precision: {precision} recall: {recall} conds: {cond_count}")
    y_pred_train = ripper_clf_A.predict(X_train)
    y_pred_test = ripper_clf_A.predict(X_test)

    df_prediction_train["Album_y_pred_train"] = y_pred_train
    df_prediction_test["Album_y_pred_test"] = y_pred_test
    return df_prediction_train, df_prediction_test


df_prediction_train, df_prediction_test = SingleTracks(
    X_train, X_test, y_train, y_test, df_prediction_train, df_prediction_test
)
df_prediction_train, df_prediction_test = LivePerformance(
    X_train, X_test, y_train, y_test, df_prediction_train, df_prediction_test
)
df_prediction_train, df_prediction_test = RadioProgram(
    X_train, X_test, y_train, y_test, df_prediction_train, df_prediction_test
)
df_prediction_train, df_prediction_test = Album(
    X_train, X_test, y_train, y_test, df_prediction_train, df_prediction_test
)


df_prediction_train.to_csv("RuleBasedResults_train_10.csv", index=False)
df_prediction_test.to_csv("RuleBasedResults_test_10.csv", index=False)
