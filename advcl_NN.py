import matplotlib.pyplot as plt
import pandas as pd
from rich import pretty, print
from rich.progress import BarColumn, Progress
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    plot_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import utils


def draw_roc(y_test, y_pred):
    lb = LabelBinarizer()
    lb.fit(y_test)
    lb.classes_.tolist()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    by_test = lb.transform(y_test)
    by_pred = lb.transform(y_pred)
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(by_test[:, i], by_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        roc_auc = roc_auc_score(by_test, by_pred, average=None)

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
    plt.title("Single Hidden Layer Neural Network Roc-Curve")
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.legend(loc="lower right", fontsize=7, frameon=False)
    plt.show()


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Neural network confusion matrix", "true"),
    ]
    # colors: Wistia too yellow
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="PuBuGn", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def execute_and_report(learn_rate, acti, current_params):
    clf = MLPClassifier(
        activation=acti,
        learning_rate_init=learn_rate,
        random_state=5213890,
        hidden_layer_sizes=current_params,
    )
    clf.fit(train_x, train_y)

    # Apply on the training set
    print("Training set:")
    y_pred = clf.predict(train_x)
    print(classification_report(train_y, y_pred))

    # Apply on the test set and evaluate the performance
    y_pred = clf.predict(test_x)
    print("Test set:")
    print(classification_report(test_y, y_pred))
    acc = accuracy_score(test_y, y_pred) * 100
    f1 = f1_score(test_y, y_pred, average="weighted") * 100

    # draw draw
    draw_confusion_matrix(clf, test_x, test_y)
    draw_roc(test_y, y_pred)
    # plt.plot(clf.loss_curve_)
    # plt.show()

    # report
    return {
        "Params": f"{acti}, {learn_rate}, {current_params}",
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
    {
        "activations": "identity",
        "learning_rate_inits": 0.001,
        "hidden_layer_sizes": (40, 40),
    },
    {
        "activations": "identity",
        "learning_rate_inits": 0.001,
        "hidden_layer_sizes": (40, 20, 8),
        # old single layer "learning_rate_inits": 0.02,
        # old single layer "hidden_layer_sizes": (40,),
    },
]
testing_params = [params[-1]]
activations = ["identity", "logistic", "tanh", "relu"]
learning_rate_inits = [0.01, 0.001, 0.02]

# progress reporting init
progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "{task.completed} of {task.total}",
)

with progress:

    # adjust len if needed
    task_layers = progress.add_task("[red]Building…", total=len(params) * 2)

    for best_params in params:
        learn_rate = best_params["learning_rate_inits"]
        acti = best_params["activations"]
        hidd = best_params["hidden_layer_sizes"]

        row = execute_and_report(learn_rate, acti, hidd)
        reports = reports.append(row, ignore_index=True)

        count += 1
        progress.advance(task_layers)

    # ------- switch up datasets: put in the 10-feature dataframe
    train_x, train_y, test_x, test_y = utils.load_tracks_xyz(
        buckets="discrete", extractclass=("album", "type"), splits=2, small=True
    ).values()

    # feature to reshape
    label_encoders = dict()
    column2encode = [
        ("track", "duration"),
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

    # rerun neural networks
    for best_params in params:
        learn_rate = best_params["learning_rate_inits"]
        acti = best_params["activations"]
        hidd = best_params["hidden_layer_sizes"]

        row = execute_and_report(learn_rate, acti, hidd)
        reports = reports.append(row, ignore_index=True)

        count += 1
        progress.advance(task_layers)
    # end switching up datasets -------

# results
print(reports.sort_values(by=["accuracy %", "F1 weighted %"], ascending=False))
print(f"I have built {count} neural networks")
