import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import pretty, print
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    plot_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

import utils


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Random Forest Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Oranges", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


pretty.install()
console = Console()
# outputs in table format
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Method", style="green")
# table.add_column("Coefficients")
table.add_column("R²", justify="right")
table.add_column("MSE", justify="right")
table.add_column("MAE", justify="right")

# DATASET
X_train, y_train, X_test, y_test = utils.load_tracks_xyz(
    buckets="discrete", extractclass=("album", "type"), splits=2
)


print(y_train.unique())

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
for col in column2encode:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.fit_transform(X_test[col])
    label_encoders[col] = le
print(X_train.info())
print(X_test.info())

le = LabelEncoder()
y_train[("album", "type")] = le.fit_transform(y_train[col])
y_test[("album", "type")] = le.fit_transform(y_test[col])

class_name = ("album", "type")

classification_report(y_true, y_pred)

"""DEEP NN-KERAS


def build_model():
    n_feature = X_train.shape[1]
    model = Sequential()
    model.add(Dense(128, input_dim=n_feature, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


model1 = build_model()

history1 = model1.fit(X_train, y_train, epochs=10, batch_size=10).history
"""
