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
    cross_val_score,
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

clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=10,
    max_features="auto",
    min_weight_fraction_leaf=0.0,
    class_weight="balanced_subsample",
    random_state=0,
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


"""Random SEARCH Random Forest

clf2 = RandomForestClassifier()
print("STA FACENDO LA RandomSEARCH")
param_list = {
    # "criterion": ["gini"],
    "max_depth": [None] + list(np.arange(40, 55)),
    # "min_samples_split": [6],
    "min_samples_leaf": list(np.arange(20, 25)),
    # "max_features": ["auto"],
    # "class_weight": ["balanced_subsample"],
}
random_search = RandomizedSearchCV(
    clf2, param_distributions=param_list, scoring="accuracy", n_iter=20, cv=5
)
res = random_search.fit(X_train, y_train)

# Print The value of best Hyperparameters
print("Best Score: %s" % res.best_score_)
print("Best Hyperparameters: %s" % res.best_params_)

print("STA FACENDO LA RandomSEARCH")
clf = RandomForestClassifier()
scores = cross_val_score(clf, X, y, cv=5)

param_list = {
    "min_samples_split": [5, 10, 15, 17, 20],
    "min_samples_leaf": [5, 10, 15, 17, 20],
}

grid_search = GridSearchCV(clf, param_grid=param_list, cv=5)
grid_search.fit(X_train, y_train)
clf = grid_search.best_estimator_

y_pred = clf.predict(X_test)

print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
print(grid_search.best_params_)
"""
