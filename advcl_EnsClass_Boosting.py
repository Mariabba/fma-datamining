from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    plot_confusion_matrix,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import utils


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Boosting  Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Greens", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


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

attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)
"""
""" """ADA BOOST DECISION TREE""" """
RISULTATI
Accuracy 0.9840317331163547
F1-score [0.99161718 0.89662028 0.970396   0.84679666]
              precision    recall  f1-score   support
           0       0.99      0.99      0.99     17183
           1       0.89      0.90      0.90       998
           2       0.97      0.97      0.97      1302
           3       0.85      0.84      0.85       181
    accuracy                           0.98     19664
   macro avg       0.93      0.93      0.93     19664
weighted avg       0.98      0.98      0.98     19664

""" """
clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(
        criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1
    ),
    n_estimators=100,
    random_state=0,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
draw_confusion_matrix(clf, X_test, y_test)

#ROC CURVE
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
plt.title("Boosting Decision Tree Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()

"""
"""BOOSTING RANDOM FOREST"""
"""
Risultati, ci mette anche lei mezz'ora
Accuracy 0.9798616761594793
F1-score [0.98922749 0.87834821 0.94351631 0.80921053]
              precision    recall  f1-score   support
           0       0.98      1.00      0.99     17183
           1       0.99      0.79      0.88       998
           2       0.98      0.91      0.94      1302
           3       1.00      0.68      0.81       181
    accuracy                           0.98     19664
   macro avg       0.99      0.84      0.91     19664
weighted avg       0.98      0.98      0.98     19664
"""
clf = AdaBoostClassifier(
    base_estimator=RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=17,
        min_samples_split=3,
        min_samples_leaf=3,
        max_features="auto",
        random_state=10,
        class_weight="balanced",
    ),
    n_estimators=100,
    random_state=0,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
draw_confusion_matrix(clf, X_test, y_test)

"""ROC CURVE"""
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
plt.title("Boosting Random Forest Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()
