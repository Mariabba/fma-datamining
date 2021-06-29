import pandas as pd
from rich import pretty, print
from rich.progress import BarColumn, Progress
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

import utils


def execute_and_report(learn_rate, acti, current_params):
    clf = MLPClassifier(
        activation=acti,
        learning_rate_init=learn_rate,
        random_state=5213890,
        **current_params,
    )
    clf.fit(train_x, train_y)

    # Apply on the test set and evaluate the performance
    y_pred = clf.predict(test_x)
    acc = accuracy_score(test_y, y_pred) * 100
    f1 = f1_score(test_y, y_pred, average="weighted") * 100
    if acc + f1 > 173:
        return {
            "Params": f"{acti}, {learn_rate}, {current_params['hidden_layer_sizes']}",
            "accuracy %": round(acc, 2),
            "F1 weighted %": round(f1, 2),
        }


pretty.install()
pd.set_option("display.max_rows", None)

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
    {"hidden_layer_sizes": (350, 350, 350)},
    {"hidden_layer_sizes": (350, 350)},
    {"hidden_layer_sizes": (350, 250, 150, 100, 50, 20)},
    {"hidden_layer_sizes": (350, 200, 100, 50, 20)},
    {"hidden_layer_sizes": (350, 200, 20)},
    {"hidden_layer_sizes": (350, 100, 20)},
    {"hidden_layer_sizes": (350, 20)},
    {"hidden_layer_sizes": (40, 40, 40, 40, 40, 40)},
    {"hidden_layer_sizes": (40, 40, 40, 40, 40)},
    {"hidden_layer_sizes": (40, 40, 40, 40)},
    {"hidden_layer_sizes": (40, 40, 40)},
    {"hidden_layer_sizes": (40, 40)},
    {"hidden_layer_sizes": (40, 33, 27, 20, 13, 8)},
    {"hidden_layer_sizes": (40, 30, 20, 10, 5)},
    {"hidden_layer_sizes": (40, 30, 20, 10)},
    {"hidden_layer_sizes": (40, 20, 10)},
    {"hidden_layer_sizes": (40, 20, 8)},
    {"hidden_layer_sizes": (40, 20)},
]
testing_params = [{"hidden_layer_sizes": (10,)}]
activations = ["identity", "relu"]
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

    for current_params in params:
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
