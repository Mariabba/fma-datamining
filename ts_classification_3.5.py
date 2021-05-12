"""CLASSIFICAZIONE CON SAX E SHAPLET RANDOM FOREST"""

"""libraries"""
from tslearn.shapelets import ShapeletModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from tslearn.piecewise import SymbolicAggregateApproximation

from music import MusicDB

from tslearn.preprocessing import TimeSeriesScalerMeanVariance

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

n_ts, ts_sz = X_train.shape
n_classes = len(set(y))
shapelet_sizes = {15: 24}

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

# Apply on the training set
print("Apply  on the training set: \n")
Y_pred = shp_clf.predict(X_train)
print("Accuracy  %s" % accuracy_score(y_train, Y_pred))
print("F1-score %s" % f1_score(y_train, Y_pred, average=None))
print(classification_report(y_train, Y_pred))

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

# 2 Best Hyperparameters: 'min_samples_split': 2, 'min_samples_leaf': 25, 'max_features': 'auto',
# 'max_depth': 22, 'criterion': 'gini', 'class_weight': None


# 3 Best Hyperparameters: {'min_samples_split': 2, 'min_samples_leaf': 30, 'max_features': 'auto',
# 'max_depth': 25, 'criterion': 'gini', 'class_weight': None}

# 1 Best Hyperparameters: {'min_samples_split': 2,
# 'min_samples_leaf': 10, 'max_features': 'auto',
# 'max_depth': 14, 'criterion': 'gini', 'class_weight': 'balanced'}
clf_rf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=30,
    max_features="auto",
    class_weight=None,
    random_state=5,
)

clf_rf.fit(X_train2, y_train)

# Apply on the training set
print("Apply  on the training set-KNN: \n")
Y_pred2 = clf_rf.predict(X_train2)
print("Accuracy  %s" % accuracy_score(y_train, Y_pred2))
print("F1-score %s" % f1_score(y_train, Y_pred2, average=None))
print(classification_report(y_train, Y_pred2))

# Apply on the test set and evaluate the performance
print("Apply on the test set and evaluate the performance-KNN: \n")
y_pred2 = clf_rf.predict(X_test2)
print("Accuracy %s" % accuracy_score(y_test, y_pred2))
print("F1-score  %s" % f1_score(y_test, y_pred2, average=None))
print(classification_report(y_test, y_pred2))

"""Random SEARCH Random Forest

clf2 = RandomForestClassifier()
print("STA FACENDO LA RandomSEARCH")
param_list = {
    "criterion": ["gini"],
    "max_depth": [None] + list(np.arange(20, 30)),
    "min_samples_split": [2],
    "min_samples_leaf": [25, 26, 28, 33, 32, 30, 35],
    "max_features": ["auto"],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
random_search = RandomizedSearchCV(
    clf2, param_distributions=param_list, scoring="accuracy", n_iter=20, cv=5
)
res = random_search.fit(X_train2, y_train)

# Print The value of best Hyperparameters
print("Best Score: %s" % res.best_score_)
print("Best Hyperparameters: %s" % res.best_params_)
"""
