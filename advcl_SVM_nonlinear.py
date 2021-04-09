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
        disp = plot_confusion_matrix(Clf, X, y, cmap="Blues", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


# DATASET
df = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
)

column2drop = [
    ("track", "language_code"),
]

df.drop(column2drop, axis=1, inplace=True)
print(df["album", "type"].unique())

# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "listens"),
    ("album", "type"),
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
"""valori con rbf test
Accuracy 0.9138018714401953
F1-score  [0.95813611 0.2641196  0.65204003 0.49795918]
              precision    recall  f1-score   support
           0       0.93      0.98      0.96     17183
           1       0.77      0.16      0.26       998
           2       0.65      0.65      0.65      1302
           3       0.95      0.34      0.50       181
    accuracy                           0.91     19664
   macro avg       0.83      0.53      0.59     19664
weighted avg       0.91      0.91      0.90     19664



valori con polynomial
Accuracy 0.9156834825061025
F1-score  [0.95881468 0.29975826 0.66126543 0.5530303 ]
              precision    recall  f1-score   support
           0       0.94      0.98      0.96     17183
           1       0.77      0.19      0.30       998
           2       0.66      0.66      0.66      1302
           3       0.88      0.40      0.55       181
    accuracy                           0.92     19664
   macro avg       0.81      0.56      0.62     19664
weighted avg       0.91      0.92      0.90     19664


valori con linear(stessa porcheria delle minear SVM)

si fa una prova con l'rbf e una con il polynomial per avere i valori besti

"""
clf = SVC(
    gamma="auto",
    kernel="poly",
)
clf.fit(X_train, y_train)

# Apply on the training set
print("Apply  on the training set: \n")
Y_pred = clf.predict(X_train)
print("Accuracy  %s" % accuracy_score(y_train, Y_pred))
print("F1-score %s" % f1_score(y_train, Y_pred, average=None))
print(classification_report(y_train, Y_pred))

# Apply on the test set and evaluate the performance
print("Apply on the test set and evaluate the performance: \n")
y_pred = clf.predict(X_test)
print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score  %s" % f1_score(y_test, y_pred, average=None))
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
plt.title("NonLinear SVM Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()
