"""CLASSIFICAZIONE  SHAPLET RANDOM FOREST DATASET COMPLETO"""

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

# librerie x shaplet
from tslearn.shapelets import ShapeletModel
from music import MusicDB

# Carico il dataframe
musi = MusicDB()
print(musi.df.info())

print(musi.feat["enc_genre"].unique())

X = musi.df
y = musi.feat["enc_genre"]  # classe targed ovvero genere con l'encoding


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("RandomForest-Shaplet Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="GnBu", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


""""CLASSIFICAZIONE 1- SHAPLET CLASSIFIER"""

scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(X).reshape(X.shape[0], X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

n_ts, ts_sz = X_train.shape
n_classes = len(set(y))

"""setting number of shaplet"""
# Set the number of shapelets per size as done in the original paper
# shapelet_sizes = grabocka_params_to_shapelet_size_dict(
#   n_ts=n_ts, ts_sz=ts_sz, n_classes=n_classes, l=0.1, r=1
# )
# risultati grabocka
# n_ts= da 7997 a 6397
# ts_size =1966
# deciso di creare invece di 8 shaplet da 269, 24 shaplet da 250
shapelet_sizes = {250: 24}

print("n_ts", n_ts)
print("ts_sz", ts_sz)
print("n_classes", n_classes)
print("shapelet_sizes", shapelet_sizes)

# Define the model using parameters provided by the authors (except that we use
# fewer iterations here)
shp_clf = ShapeletModel(
    n_shapelets_per_size=shapelet_sizes,
    optimizer="sgd",
    weight_regularizer=0.01,
    max_iter=50,
    verbose=1,
)

shp_clf.fit(X_train, y_train)

# Apply on the test set and evaluate the performance
print("Apply on the test set and evaluate the performance: \n")
y_pred = shp_clf.predict(X_test)
print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score  %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))


""" SHAPLET BASED Random Forest"""

X_train2 = shp_clf.transform(X_train)
print("train shape:", X_train2.shape)
X_test2 = shp_clf.transform(X_test)
# Best: {'random_state': 2, 'min_samples_split': 3, 'min_samples_leaf': 20,
# 'max_features': 'log2', 'max_depth': 5, 'criterion': 'gini', 'class_weight': None}

# Best VERA: {'random_state': 2, 'min_samples_split': 3, 'min_samples_leaf': 50,
# 'max_features': 'log2', 'max_depth': 14, 'criterion': 'gini',
# 'class_weight': 'balanced_subsample'}

clf_rf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=14,
    min_samples_split=3,
    min_samples_leaf=50,
    max_features="log2",
    class_weight="balanced_subsample",
    random_state=2,
)

clf_rf.fit(X_train2, y_train)

# Apply on the test set and evaluate the performance
print("Apply on the test set and evaluate the performance-KNN: \n")
y_pred2 = clf_rf.predict(X_test2)
print("Accuracy %s" % accuracy_score(y_test, y_pred2))
print("F1-score  %s" % f1_score(y_test, y_pred2, average=None))
print(classification_report(y_test, y_pred2))

draw_confusion_matrix(clf_rf, X_test2, y_test)

"""ROC CURVE"""
lb = LabelBinarizer()
lb.fit(y_test)
lb.classes_.tolist()

fpr = dict()
tpr = dict()
roc_auc = dict()
by_test = lb.transform(y_test)
by_pred = lb.transform(y_pred2)
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
plt.title("Random Forest-Shaplet Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()

"""GRID SEARCH SHAPLET BASED Random Forest

clf = RandomForestClassifier()
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
random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=20, cv=5)
random_search.fit(X_train2, y_train)

# Print The value of best Hyperparameters
print(
    "Best:",
    random_search.cv_results_["params"][
        random_search.cv_results_["rank_test_score"][0]
    ],
)
"""
