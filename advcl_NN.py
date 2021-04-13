import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import pretty, print
from rich.console import Console
from rich.table import Table
from rich.progress import track
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    plot_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

import utils


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Neural network confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Blues", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def execute_and_report(learn_rate, acti, current_params):
    clf = MLPClassifier(
        activation=acti,
        learning_rate_init=learn_rate,
        random_state=5213890,
        verbose=1,
        **current_params,
    )
    clf.fit(train_x, train_y)

    # Apply on the training set
    # print("Training set:")
    # Y_pred = clf.predict(train_x)
    # print(classification_report(train_y, Y_pred))

    # Apply on the test set and evaluate the performance
    # print("Test set: \n")
    y_pred = clf.predict(test_x)
    reports[f"{acti}, {learn_rate}, {str(current_params.values())}"] = [
        f1_score(test_y, y_pred, average="weighted"),
        accuracy_score(test_y, y_pred),
    ]

    # draw_confusion_matrix(clf, test_x, test_y)

    # plt.plot(clf.loss_curve_)
    # plt.show()


pretty.install()
console = Console()
# outputs in table format
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Method", style="green")
# table.add_column("Coefficients")
table.add_column("Accuracy", justify="right")
table.add_column("Precision", justify="right")
table.add_column("Recall", justify="right")
table.add_column("F1", justify="right")

# DATASET
train_x, train_y, test_x, test_y = utils.load_tracks_xyz(
    buckets="discrete", extractclass=("album", "type"), splits=2
).values()

# feature to reshape
label_encoders = dict()
column2encode = [
    ("track", "language_code"),
    ("album", "listens"),
    ("track", "license"),
    ("album", "comments"),
    ("album", "date_created"),
    ("album", "favorites"),
    ("artist", "comments"),
    ("artist", "date_created"),
    ("artist", "favorites"),
    ("track", "comments"),
    ("track", "date_created"),
    ("track", "duration"),
    ("track", "favorites"),
    ("track", "interest"),
    ("track", "listens"),
]
for col in track(column2encode):
    le = LabelEncoder()
    le.fit(test_x[col])
    train_x[col] = le.fit_transform(train_x[col])
    test_x[col] = le.fit_transform(test_x[col])
    label_encoders[col] = le

le = LabelEncoder()
le.fit(train_y)
test_y = le.fit_transform(test_y)
train_y = le.fit_transform(train_y)

class_name = ("album", "type")

# Preparation
reports = {}
params = [
    # {"hidden_layer_sizes": (150,)},
    # {"hidden_layer_sizes": (100,)},
    # {"hidden_layer_sizes": (60,)},
    # {"hidden_layer_sizes": (30,)},
    {"hidden_layer_sizes": (10,)},
]
activations = ["identity", "logistic", "tanh", "relu"]
learning_rate_inits = [0.01, 0.001, 0.02]

for i, current_params in enumerate(params):
    for learn_rate in learning_rate_inits:
        for acti in activations:
            print(f"[magenta]Run number[/magenta] {i}")
            execute_and_report(learn_rate, acti, current_params)

print(reports)
