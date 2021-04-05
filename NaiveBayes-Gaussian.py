import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    plot_confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

import utils


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Gaussian Naive Bayes Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Purples", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


"""
df = utils.load_tracks_xyz(buckets="continuous", extractclass=("album", "type"))
# train_x, train_y, vali_x, vali_y, test_x, test_y
clf = GaussianNB()
clf.fit(df["train_x"], df["train_y"])
print(clf)
# y_pred = clf.predict(df["vali_x"])
# print("Accuracy %s" % accuracy_score(df["vali_y"], y_pred))
# print("F1-score %s" % f1_score(y_test, y_pred, average=None))
# print(classification_report(y_test, y_pred))
"""
df = utils.load(
    "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
)

column2drop = [
    ("album", "title"),
    ("artist", "name"),
    ("set", "split"),
    ("track", "title"),
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "language_code"),
    ("track", "number"),
    ("track", "tags"),
    ("track", "genres"),
    ("track", "genres_all"),
]
df.drop(column2drop, axis=1, inplace=True)
df = df[df["album", "type"] != "Contest"]

# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "listens"),
    ("album", "type"),
    ("track", "license"),
    ("album", "id"),
    ("album", "comments"),
    ("album", "date_created"),
    ("album", "favorites"),
    ("album", "tracks"),
    ("artist", "comments"),
    ("artist", "date_created"),
    ("artist", "favorites"),
    ("artist", "id"),
    ("track", "comments"),
    ("track", "date_created"),
    ("track", "duration"),
    ("track", "favorites"),
    ("track", "interest"),
    ("track", "listens"),
]
for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
df.info()
class_name = ("album", "type")
"""
print("LISTA COLONNE")
column_names = list(df.columns)
for column_name in column_names:
    unique_values = df[column_name].unique()
    if len(unique_values) >= 10:
        print(column_name, "more than 10 %s values" % df.dtypes[column_name], sep="\t")
    else:
        print(column_name, unique_values, sep="\t")
"""
attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

"""NB GAUSSIAN"""
clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
draw_confusion_matrix(clf, X_test, y_test)

"""ROC Curve"""

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
plt.title("Gaussian Naive Bayes Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()
