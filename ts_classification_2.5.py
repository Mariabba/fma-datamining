"""CLASSIFICAZIONE RANDOM FOREST DATASET COMPLETO"""

"""libraries"""
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    plot_confusion_matrix,
)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.preprocessing import LabelBinarizer
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from music import MusicDB

# Carico Dataframe

musi = MusicDB()
print(musi.df.info())


print(musi.feat["enc_genre"].unique())

X = musi.df
y = musi.feat["enc_genre"]

scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(X).reshape(X.shape[0], X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("RandomForest-TimeSeries Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="GnBu", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


"""Random Forest sulle TS"""
# Best PRIMA VOLTA: {'random_state': 5, 'min_samples_split': 50, 'min_samples_leaf': 5,
# 'max_features': 'log2', 'max_depth': 13, 'criterion': 'gini',
# 'class_weight': 'balanced'}

# Best a 50
# Best Hyperparameters: {'random_state': 5, 'min_samples_split': 100,
# 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 17,
# 'criterion': 'gini', 'class_weight': 'balanced'}

# Best a 20
# Best Hyperparameters: {'random_state': 5, 'min_samples_split': 3,
# 'min_samples_leaf': 30, 'max_features': 'log2', 'max_depth': 9,
# 'criterion': 'gini', 'class_weight': 'balanced_subsample'}

clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=13,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="log2",
    class_weight="balanced_subsample",
    random_state=5,
)
clf.fit(X_train, y_train)


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
for i in range(8):
    fpr[i], tpr[i], _ = roc_curve(by_test[:, i], by_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    roc_auc = roc_auc_score(by_test, by_pred, average=None)

plt.figure(figsize=(8, 5))
for i in range(8):
    plt.plot(
        fpr[i],
        tpr[i],
        label="%s ROC curve (area = %0.2f)" % (lb.classes_.tolist()[i], roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title("RandomForest-TimeSeries Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()

"""Random SEARCH Random Forest

clf2 = RandomForestClassifier()
print("STA FACENDO LA RandomSEARCH")
param_list = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None] + list(np.arange(2, 20)),
    "min_samples_split": [2, 3, 5, 7, 10, 20, 30, 50, 100],
    "min_samples_leaf": [1, 3, 5, 10, 20, 30, 50, 100],
    "max_features": ["auto", "sqrt", "log2"],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "random_state": [0, 2, 5, 10],
}
random_search = RandomizedSearchCV(
    clf2, param_distributions=param_list, scoring="accuracy", n_iter=20, cv=5
)
res = random_search.fit(X_train, y_train)

# Print The value of best Hyperparameters
print("Best Score: %s" % res.best_score_)
print("Best Hyperparameters: %s" % res.best_params_)
"""
