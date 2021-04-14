import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    plot_confusion_matrix,
)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from collections import defaultdict
import utils
import wittgenstein as lw

from sklearn.model_selection import train_test_split


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Rules Based Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Reds", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


class_name = ("album", "type")

df = utils.load_tracks(buckets="discrete")
# df=df.head(1000)
column2drop = [
    ("track", "license"),
    ("track", "language_code"),
]
df.drop(column2drop, axis=1, inplace=True)

print(df.info())
df["album", "type"] = df["album", "type"].replace(
    ["Single Tracks", "Live Performance", "Radio Program"],
    ["NotAlbum", "NotAlbum", "NotAlbum"],
)

df["album", "type"] = df["album", "type"].replace(["Album", "NotAlbum"], [True, False])

"""
# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "type"),
]

for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
"""

attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep="=")
dfY = df[class_name]
df = pd.concat([dfX, dfY], axis=1)
print(df.info())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

ripper_clf = lw.RIPPER()
"""
ripper_clf.fit(X_train, y_train, class_feat=("album", "type"), pos_class=0, k=1, prune_size=0.33)
print(ripper_clf)

# Collect performance metrics
precision = ripper_clf.score(X_test, y_test, precision_score)
recall = ripper_clf.score(X_test, y_test, recall_score)
cond_count = ripper_clf.ruleset_.count_conds()
print(ripper_clf.ruleset_.out_pretty())
print(f'precision: {precision} recall: {recall} conds: {cond_count}')




# Apply the decision tree on the training set
print("Apply the decision tree on the training set: \n")
y_pred = ripper_clf.predict(X_train)
y_train = y_train.apply(lambda x: 1 - x)
print("Accuracy %s" % accuracy_score(y_train, y_pred))
print("F1-score %s" % f1_score(y_train, y_pred, average=None))
print(classification_report(y_train, y_pred))
confusion_matrix(y_train, y_pred)

# Apply the decision tree on the test set and evaluate the performance
print("Apply the decision tree on the test set and evaluate the performance: \n")
y_pred = ripper_clf.predict(X_test)
y_test = y_test.apply(lambda x: 1 - x)
print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
draw_confusion_matrix(ripper_clf, X_test, y_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)

roc_auc = roc_auc_score(y_test, y_pred, average=None)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.legend(loc="lower right", fontsize=14, frameon=False)
plt.show()


"""
print("GRID SEARCH:")
# grid search
param_grid = {
    "prune_size": [0.33, 0.5],
    "k": [1, 2],
    "class_feat": ["album", "type"],
    "pos_class": [0],
}
grid = GridSearchCV(estimator=ripper_clf, param_grid=param_grid)
grid.fit(X_train, y_train)
clf = grid.best_estimator_
print(report(grid.cv_results_, n_top=3))
