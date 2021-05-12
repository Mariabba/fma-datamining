"""CLASSIFCAZIONE CON SAX E RANDOM FOREST"""
"""libraries"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
)
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from music import MusicDB

"""FILE 4  CLASSIFICAZIONE CON SAX RANDOM FOREST"""


# Carico il dataframe
musi = MusicDB()
print(musi.df.info())

print(musi.feat["enc_genre"].unique())


X_no = musi.df
y = musi.feat["enc_genre"]  # classe targed ovvero genere con l'encoding

# normalizzazione con mean variance
scaler = TimeSeriesScalerMeanVariance()
X_no = pd.DataFrame(
    scaler.fit_transform(musi.df.values).reshape(
        musi.df.values.shape[0], musi.df.values.shape[1]
    )
)
X_no.index = musi.df.index

# approssimazione con sax

sax = SymbolicAggregateApproximation(n_segments=130, alphabet_size_avg=20)
X1 = sax.fit_transform(X_no)
print(X1.shape)
X = np.squeeze(X1)
print(X.shape)

# Classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)


"""Random Forest sulle TS"""
# Best PRIMA VOLTA: {'random_state': 5, 'min_samples_split': 50, 'min_samples_leaf': 5,
# 'max_features': 'log2', 'max_depth': 13, 'criterion': 'gini',
# 'class_weight': 'balanced'}

# Best a 50
# Best Hyperparameters: {'random_state': 5, 'min_samples_split': 100,
# 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 17,
# 'criterion': 'gini', 'class_weight': 'balanced'}


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

# todo ricorda mari, manca solo questa
"""Random SEARCH Random Forest

clf = RandomForestClassifier()
print("STA FACENDO LA RandomSEARCH")
param_list = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None] + list(np.arange(2, 20)),
    "min_samples_split": [2, 3, 5, 7, 10, 20, 30, 50, 100],
    "min_samples_leaf": [1, 3, 5, 10, 20, 30, 50, 100],
    "max_features": ["auto", "sqrt", "log2"],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
grid_search = GridSearchCV(clf, param_grid=param_list, scoring="accuracy", cv=5)
grid_search.fit(X_train, y_train)

# results of the grid search
print("\033[1m" "Results of the grid search" "\033[0m")
print()
print("Best parameters: %s" % grid_search.best_params_)
print("Best estimator: %s" % grid_search.best_score_)
print()
print("Best k ('n_neighbors'): %s" % grid_search.best_params_["n_neighbors"])
print()
"""
