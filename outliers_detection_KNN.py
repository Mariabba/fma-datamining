import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# sklearn
from pyod.utils.example import visualize
from sklearn import metrics
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    StandardScaler,
    LabelEncoder,
)
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    make_scorer,
    precision_recall_curve,
)
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier

from pandas import DataFrame

from pandas import DataFrame
import utils
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from pyod.models.knn import KNN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# DATASET
df = utils.load_tracks(buckets="continuous", outliers=False)

column2drop = [
    ("track", "language_code"),
    ("track", "license")
]

df.drop(column2drop, axis=1, inplace=True)
print(df.info())
"""
def normalize(feature):
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])


for col in df.columns:
    normalize(col)
"""

"""
# FACCIO  IL PLOTTING BOXPLOT del Df completo
plt.figure(figsize=(20, 25))
b = sns.boxplot(data=df, orient="h")
b.set(ylabel="Class", xlabel="Normalization Value")
plt.show()
"""
# X = df.drop(columns=[("album", "type")])
# y = df[("album", "type")]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

#X = df.values

"""
# instantiate model
nbrs = NearestNeighbors(n_neighbors = 3)
# fit model
nbrs.fit(X)

# distances and indexes of k-neaighbors from model outputs
distances, indexes = nbrs.kneighbors(X)
# plot mean of k-distances of each observation
plt.plot(distances.mean(axis =1))
plt.show()

# visually determine cutoff values > 10
outlier_index = np.where(distances.mean(axis = 1) > 0.8)
print(outlier_index)

# filter outlier values
outlier_values = df.iloc[outlier_index]
print(outlier_values.describe())
print(outlier_values.value_counts())
print(outlier_values.info())
"""


"""
# train kNN detector
clf_name = 'KNN'
clf = KNN(n_neighbors=2)
clf.fit(X)
# If you want to see the predictions of the training data, you can use this way:
y_train_scores = clf.decision_scores_
print("Prediction on the training data: ")
print(y_train_scores)

# Now we have the trained K-NN model, let's apply to the test data to get the predictions
y_test_pred = clf.predict(X) # outlier labels (0 or 1)
# Because it is '0' and '1', we can run a count statistic.
unique, counts = np.unique(y_test_pred, return_counts=True)
print("Prediction on the test data: ")
print(dict(zip(unique, counts)))

# And you can generate the anomaly score using clf.decision_function:
y_test_scores = clf.decision_function(X)
print("Anomaly score: ")
plt.hist(y_test_scores, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()

print("Anomaly mean for outliers: ")
print(np.mean(y_test_scores[np.where(y_test_pred==1)]))
print("Anomaly mean for inliers: ")
print(np.mean(y_test_scores[np.where(y_test_pred==0)]))

print(np.mean(clf.decision_scores_[np.where(y_test_pred==1)]))
print(np.mean(clf.decision_scores_[np.where(y_test_pred==0)]))

plt.hist(clf.decision_scores_, bins=20)
plt.axvline(np.min(clf.decision_scores_[np.where(y_test_pred==1)]), c='k')
plt.show()
"""

"""
def tuning_param(df):
    print(df.info)
    print(df.head)
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=100)
    print(X_train, X_test)
    from pyod.models.combination import aom, moa, average, maximization
    from pyod.utils.utility import standardizer

    # Standardize data
    X_train_norm, X_test_norm = standardizer(X_train, X_test)
    # Test a range of k-neighbors from 10 to 200. There will be 20 k-NN models.
    n_clf = 5
    k_list = [2, 5, 10, 20, 50]
    # Just prepare data frames so we can store the model results
    train_scores = np.zeros([X_train.shape[0], n_clf])
    test_scores = np.zeros([X_test.shape[0], n_clf])
    print(train_scores.shape)
    # Modeling
    for i in range(n_clf):
        k = k_list[i]
        clf = KNN(n_neighbors=k, method="largest")
        clf.fit(X_train_norm)

        # Store the results in each column:
        train_scores[:, i] = clf.decision_scores_
        test_scores[:, i] = clf.decision_function(X_test_norm)
        # Decision scores have to be normalized before combination
    train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

    # Combination by average
    # The test_scores_norm is 500 x 20. The "average" function will take the average of the 20 columns. The result "y_by_average" is a single column:
    y_by_average = average(test_scores_norm)
    import matplotlib.pyplot as plt

    plt.hist(y_by_average, bins="auto")  # arguments are passed to np.histogram
    plt.title("Combination by average")
    plt.show()

    df_test = pd.DataFrame(X_test)
    df_test["y_by_average_score"] = y_by_average
    df_test["y_by_average_cluster"] = np.where(df_test["y_by_average_score"] < 0, 0, 1)
    print(df_test["y_by_average_cluster"].value_counts())


tuning_param(df)
"""