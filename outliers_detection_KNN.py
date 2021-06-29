import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import utils

# DATASET
df = utils.load_tracks(buckets="continuous", outliers=False)

column2drop = [
    ("track", "language_code"),
    ("track", "license"),
    ("artist", "wikipedia_page"),
    ("track", "composer"),
    ("track", "information"),
    ("track", "lyricist"),
    ("track", "publisher"),
    ("album", "engineer"),
    ("album", "information"),
    ("artist", "bio"),
    ("album", "producer"),
    ("artist", "website"),
]

df.drop(column2drop, axis=1, inplace=True)
print(df.info())


def normalize(feature):
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])


colum2encode = [col for col in df.columns if col != ("album", "type")]
for col in colum2encode:
    normalize(col)

"""
# FACCIO  IL PLOTTING BOXPLOT del Df completo
plt.figure(figsize=(20, 25))
b = sns.boxplot(data=df, orient="h")
b.set(ylabel="Class", xlabel="Normalization Value")
plt.show()
"""
X = df.drop(columns=[("album", "type")])
y = df[("album", "type")]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

# instantiate model
nbrs = NearestNeighbors(n_neighbors=5, p=1)
# fit model
nbrs.fit(X)

# distances and indexes of k-neighbors from model outputs
distances, indexes = nbrs.kneighbors(X)
# plot mean of k-distances of each observation
plt.ylabel("k-neaighbors distance")
plt.plot(distances.mean(axis=1), color="black")
plt.axhline(y=1.5, color="r", linestyle="-")
plt.show()

# visually determine cutoff values > 10 3.8
outlier_index = np.where(distances.mean(axis=1) > 2.5)

# filter outlier values
# outlier_values = df.iloc[outlier_index]
# print(outlier_values.describe())
# print(outlier_values.value_counts())
# print(outlier_values.info())

outlier_values = df.iloc[outlier_index]

df["Outlier"] = df.index.isin(outlier_values.index)
knn_outliers = df["Outlier"]
knn_outliers.to_csv("strange_results_new/KNN.csv")

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
