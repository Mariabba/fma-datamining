from matplotlib import pyplot

import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Rules Based Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="RdPu", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


class_name = ("album", "type")


df = utils.load_tracks("data/tracks.csv", outliers=True, buckets="continuous")
df["album", "type"] = df["album", "type"].replace(
    ["Single Tracks", "Live Performance", "Radio Program"],
    ["NotAlbum", "NotAlbum", "NotAlbum"],
)

column2drop = [
    ("track", "language_code"),
    ("track", "license"),
]

df.drop(column2drop, axis=1, inplace=True)


# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "type"),
]


for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


def normalize(feature):
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])


colum2encode = [col for col in df.columns if col not in [("album", "type")]]
for col in colum2encode:
    normalize(col)

print(df.info())

attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)
clf = LogisticRegression(random_state=0, max_iter=1000, C=0.1, penalty="l2")
clf.fit(X_train, y_train)

# get importance
importance = clf.coef_[0]


# test
top_n = 10
feat_imp = pd.DataFrame(columns=["columns", "importance"])
for col, imp in zip(attributes, importance):
    feat_imp = feat_imp.append({"columns": col, "importance": imp}, ignore_index=True)
print(feat_imp)

feat_imp = feat_imp.reindex(
    feat_imp.importance.abs().sort_values(ascending=False).index
)
feat_imp = feat_imp.iloc[:top_n]

feat_imp.plot(
    title="Top 10 coefficient of the features in the decision function",
    x="columns",
    fontsize=8.4,
    rot=15,
    y="importance",
    kind="bar",
    colormap="PiYG",
)
plt.axhline(y=0, color="b", linestyle="-")
plt.show()

# test


"""
n = -2
for col in df.columns:
    n = n + 1
    plt.scatter(X_train[:, n], y_train)

    print("ok")
    plt.xlabel(col, fontsize=16)
    plt.ylabel('Album = 0 NotAlbum = 1', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()
"""
# Apply the decision tree on the training set
print("Apply the decision tree on the training set: \n")
y_pred = clf.predict(X_train)
print("Accuracy %s" % accuracy_score(y_train, y_pred))
print("F1-score %s" % f1_score(y_train, y_pred, average=None))
print(classification_report(y_train, y_pred))

confusion_matrix(y_train, y_pred)

# Apply the decision tree on the test set and evaluate the performance
print("Apply the decision tree on the test set and evaluate the performance: \n")

y_pred = clf.predict(X_test)

print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
draw_confusion_matrix(clf, X_test, y_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)

roc_auc = roc_auc_score(y_test, y_pred, average=None)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % (roc_auc))

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)
plt.tick_params(axis="both", which="major", labelsize=22)
plt.legend(loc="lower right", fontsize=14, frameon=False)
plt.show()


"""
n = -2
for col in df.columns:
    n +=1
    loss = expit(sorted(X_test[:, n]) * clf.coef_[:, n] + clf.intercept_).ravel()
    plt.plot(sorted(X_test[:, n]), loss, color='red', linewidth=3)
    plt.scatter(X_train[:, n], y_train)
    plt.xlabel(col, fontsize=16)
    plt.ylabel('Album Type', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()
"""

"""
print("GRID SEARCH:")
# Grid search cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-3,3,7), "penalty":['none' ,"l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression(random_state=0, max_iter=1000)
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
"""
