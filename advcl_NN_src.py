import matplotlib.pyplot as plt
import pandas as pd
from rich import pretty, print
from rich.progress import Progress, BarColumn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    plot_confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

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
    acc = accuracy_score(test_y, y_pred) * 100
    f1 = f1_score(test_y, y_pred, average="weighted") * 100
    if acc + f1 > 170:
        return {
            "Params": f"{acti}, {learn_rate}, {current_params['hidden_layer_sizes'][0]}",
            "accuracy %": round(acc, 2),
            "F1 weighted %": round(f1, 2),
        }

    # draw_confusion_matrix(clf, test_x, test_y)

    # plt.plot(clf.loss_curve_)
    # plt.show()


pretty.install()

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
for col in column2encode:
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
count = 0
reports = pd.DataFrame(columns=["Params", "accuracy %", "F1 weighted %"])
params = [
    {"hidden_layer_sizes": (500,)},
    {"hidden_layer_sizes": (450,)},
    {"hidden_layer_sizes": (400,)},
    {"hidden_layer_sizes": (350,)},
    {"hidden_layer_sizes": (300,)},
    {"hidden_layer_sizes": (260,)},
    {"hidden_layer_sizes": (220,)},
    {"hidden_layer_sizes": (180,)},
    {"hidden_layer_sizes": (150,)},
    {"hidden_layer_sizes": (120,)},
    {"hidden_layer_sizes": (100,)},
    {"hidden_layer_sizes": (80,)},
    {"hidden_layer_sizes": (65,)},
    {"hidden_layer_sizes": (50,)},
    {"hidden_layer_sizes": (40,)},
    {"hidden_layer_sizes": (30,)},
    {"hidden_layer_sizes": (20,)},
    {"hidden_layer_sizes": (10,)},
]
testing_params = [{"hidden_layer_sizes": (10,)}]
activations = ["identity", "logistic", "tanh", "relu"]
learning_rate_inits = [0.01, 0.001, 0.02]


# progress reporting init
progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "{task.completed} of {task.total}",
)

# grid search
with progress:

    task_layers = progress.add_task("[red]Hidden layer sizes…", total=len(params))
    task_learn = progress.add_task("[green]Learn rate…", total=len(learning_rate_inits))
    task_acti = progress.add_task("[cyan]Activation funcs…", total=len(activations))

    for current_params in testing_params:
        progress.update(task_learn, completed=0)
        for learn_rate in learning_rate_inits:
            progress.update(task_acti, completed=0)
            for acti in activations:
                row = execute_and_report(learn_rate, acti, current_params)
                if row:
                    reports = reports.append(row, ignore_index=True)
                count += 1
                progress.advance(task_acti)
            progress.advance(task_learn)
        progress.advance(task_layers)

# results
print(reports.sort_values(by=["accuracy %", "F1 weighted %"], ascending=False))
print(f"I have built {count} neural networks")
