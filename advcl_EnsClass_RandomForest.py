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
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC, SVC

import utils


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Random Forest Confusion matrix", "true"),
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
# STANDARDIZZO
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
"""
"""RANDOM FOREST"""
# 1 grid search:Best: {'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 17}
# 2 Best: {'random_state': 10, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_depth': 17,
# 'criterion': 'gini', 'class_weight': 'balanced'}
clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=17,
    min_samples_split=3,
    min_samples_leaf=3,
    max_features="auto",
    random_state=10,
    class_weight="balanced",
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
plt.title("Random Forest Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()

"""Feature Importance"""
nbr_features = 30

tree_feature_importances = clf.feature_importances_
sorted_idx = tree_feature_importances.argsort()[-nbr_features:]

y_ticks = np.arange(0, len(sorted_idx))
fig, ax = plt.subplots()
plt.figure(figsize=(15, 10))
plt.barh(y_ticks, tree_feature_importances[sorted_idx])
plt.yticks(y_ticks, attributes)
plt.ylabel("class name", fontsize=7)

plt.title("Random Forest Feature Importances (MDI)")
plt.show()

"""Permutation Importance"""
result = permutation_importance(
    clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)

sorted_idx = result.importances_mean.argsort()[-nbr_features:]

fig, ax = plt.subplots()
plt.figure(figsize=(15, 10))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=attributes)
plt.title("Permutation Importances (test set)")
plt.tight_layout()
plt.show()
"""
print("STA FACENDO LA GRIDSEARCH")
param_list = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None] + list(np.arange(2, 20)),
    "min_samples_split": [2, 3, 5, 7, 10, 20, 30, 50, 100],
    "min_samples_leaf": [1, 3, 5, 10, 20, 30, 50, 100],
    # "max_features": ["auto", "sqrt", "log2"],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "random_state": [0, 2, 5, 10],
}

random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=20, cv=5)
random_search.fit(X_train, y_train)
clf = random_search.best_estimator_

y_pred = clf.predict(X_test)
# Print The value of best Hyperparameters
print(
    "Best:",
    random_search.cv_results_["params"][
        random_search.cv_results_["rank_test_score"][0]
    ],
)
"""
