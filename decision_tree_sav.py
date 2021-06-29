import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    plot_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import utils

os.environ["PATH"] += (
    os.pathsep
    + "C:/Users/saverio/Desktop/Data Mining/DataMiningProject/venvv/Lib/site-packages/graphviz/bin"
)


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Decision Tree Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="OrRd", normalize=normalize)
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


def load_data(path):
    dfs = utils.load_tracks_xyz(
        buckets="discrete", splits=2, extractclass=("album", "type"), outliers=False
    )
    # feature to reshape
    label_encoders = dict()
    column2encode = [
        ("album", "comments"),
        ("album", "favorites"),
        ("album", "listens"),
        ("artist", "comments"),
        ("artist", "favorites"),
        ("track", "duration"),
        ("track", "comments"),
        ("track", "favorites"),
        ("track", "language_code"),
        ("track", "license"),
        ("track", "listens"),
    ]

    for col in column2encode:
        le = LabelEncoder()
        dfs["train_x"][col] = le.fit_transform(dfs["train_x"][col])
        dfs["test_x"][col] = le.fit_transform(dfs["test_x"][col])
        label_encoders[col] = le

    le1 = LabelEncoder()
    dfs["train_y"] = le1.fit_transform(dfs["train_y"])
    dfs["test_y"] = le1.fit_transform(dfs["test_y"])
    label_encoders[("album", "type")] = le1
    return dfs


def tuning_param(target1, target2):
    df = utils.load_tracks("data/tracks.csv", outliers=False)
    # feature to reshape
    label_encoders = dict()
    column2encode = [
        ("album", "comments"),
        ("album", "favorites"),
        ("album", "listens"),
        ("album", "type"),
        ("artist", "comments"),
        ("artist", "favorites"),
        ("track", "duration"),
        ("track", "comments"),
        ("track", "favorites"),
        ("track", "language_code"),
        ("track", "license"),
        ("track", "listens"),
    ]

    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print(df.info())

    # split dataset train and set
    attributes = [col for col in df.columns if col != (target1, target2)]
    X = df[attributes].values
    y = df[target1, target2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    print(X_train.shape, X_test.shape)

    # tuning hyperparam with randomize search

    # This has two main benefits over an exhaustive search:

    # A budget can be chosen independent of the number of parameters and possible values.

    # Adding parameters that do not influence the performance does not decrease efficiency.
    # RANDOM
    print("Parameter Tuning: \n")

    # tuning parameters with random search
    print("Search best parameters: \n")
    param_list = {
        "max_depth": [None] + list(np.arange(2, 50)),
        "min_samples_split": [2, 5, 10, 15, 20, 30, 50, 100, 150],
        "min_samples_leaf": [1, 2, 5, 10, 15, 20, 30, 50, 100, 150],
        "criterion": ["gini", "entropy"],
    }

    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1
    )

    random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=100)
    random_search.fit(X, y)
    report(random_search.cv_results_, n_top=10)


def build_model(
    df, target1, target2, min_samples_split, min_samples_leaf, max_depth, criterion
):

    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    clf.fit(df["train_x"], df["train_y"])
    # value importance

    # split dataset train and set
    dfs = utils.load_tracks("data/tracks.csv", outliers=False, buckets="discrete")
    attributes = [col for col in dfs.columns if col != (target1, target2)]
    X = dfs[attributes].values
    y = dfs[target1, target2]

    for col, imp in zip(attributes, clf.feature_importances_):
        print(col, imp)

    top_n = 10
    feat_imp = pd.DataFrame(columns=["columns", "importance"])
    for col, imp in zip(attributes, clf.feature_importances_):
        feat_imp = feat_imp.append(
            {"columns": col, "importance": imp}, ignore_index=True
        )
    print(feat_imp)

    feat_imp.sort_values(by="importance", ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]

    feat_imp.plot(
        title="Top 10 features contribution",
        x="columns",
        fontsize=8.5,
        rot=15,
        y="importance",
        kind="bar",
        colormap="Pastel1",
    )
    plt.show()

    """
    pyplot.bar(
        [x for x in range(len(clf.feature_importances_))], clf.feature_importances_
    )
    pyplot.show()
    """
    # Apply the decision tree on the training set
    print("Apply the decision tree on the training set: \n")
    y_pred = clf.predict(df["train_x"])
    print("Accuracy %s" % accuracy_score(df["train_y"], y_pred))
    print("F1-score %s" % f1_score(df["train_y"], y_pred, average=None))

    print(classification_report(df["train_y"], y_pred))

    confusion_matrix(df["train_y"], y_pred)

    # Apply the decision tree on the test set and evaluate the performance
    print("Apply the decision tree on the test set and evaluate the performance: \n")
    y_pred = clf.predict(df["test_x"])

    print(classification_report(df["test_y"], y_pred))

    print("\033[1m" "Metrics" "\033[0m")

    print("Accuracy %s" % accuracy_score(df["test_y"], y_pred))
    print("F1-score %s" % f1_score(df["test_y"], y_pred, average=None))

    confusion_matrix(df["test_y"], y_pred)

    # ROC Curve
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    lb.fit(df["test_y"])
    lb.classes_.tolist()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    by_test = lb.transform(df["test_y"])
    by_pred = lb.transform(y_pred)
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(by_test[:, i], by_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    roc_auc = roc_auc_score(by_test, by_pred, average=None)
    print(roc_auc)

    plt.figure(figsize=(8, 5))
    for i in range(4):
        plt.plot(
            fpr[i],
            tpr[i],
            label="%s ROC curve (area = %0.2f)" % (lb.classes_.tolist()[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=22)
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.show()

    # Model Accuracy, how often is the classifier correct?
    draw_confusion_matrix
    print("Accuracy:", metrics.accuracy_score(df["test_y"], y_pred))
    # confusion matrix
    print("\033[1m" "Confusion matrix" "\033[0m")

    draw_confusion_matrix(clf, df["test_x"], df["test_y"])

    print()


tracks = load_data("data/tracks.csv")
# tuning_param("album", "type")

# build_model(tracks, "album", "type", 100, 100, 8, "entropy")
# build_model(tracks, "album", "type", 2, 1, 20, "entropy")
# build_model(tracks, "album", "type", 20, 100, 20, "entropy")
# build_model(tracks, "album", "type", 20, 20, 9, "gini")
build_model(tracks, "album", "type", 10, 10, 9, "gini")
