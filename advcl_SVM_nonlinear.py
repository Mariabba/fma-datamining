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
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC, SVC

import utils


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("NonLinear SVM Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Purples", normalize=normalize)
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
# STANDARDIZZO
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

"""NON LINEAR SVM CLASSIFIER"""
"""valori con rbf
Accuracy 0.9167005695687551
F1-score [0.95982775 0.30219334 0.65465936 0.52244898]
              precision    recall  f1-score   support
           0       0.94      0.99      0.96     17183
           1       0.80      0.19      0.30       998
           2       0.67      0.64      0.65      1302
           3       1.00      0.35      0.52       181
    accuracy                           0.92     19664
   macro avg       0.85      0.54      0.61     19664
weighted avg       0.91      0.92      0.90     19664

valori con polynomial
Accuracy 0.9203620829943043
F1-score [0.96153082 0.32885375 0.67344544 0.67137809]
              precision    recall  f1-score   support
           0       0.94      0.99      0.96     17183
           1       0.78      0.21      0.33       998
           2       0.69      0.66      0.67      1302
           3       0.93      0.52      0.67       181
    accuracy                           0.92     19664
   macro avg       0.83      0.60      0.66     19664
weighted avg       0.91      0.92      0.91     19664

valori con linear(stessa porcheria delle minear SVM)

si fa una prova con l'rbf e una con il polynomial per avere i valori besti

"""
clf = SVC(
    gamma="auto",
    kernel="rbf",
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
draw_confusion_matrix(clf, X_test, y_test)


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
plt.title("NonLinear SVM Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()
