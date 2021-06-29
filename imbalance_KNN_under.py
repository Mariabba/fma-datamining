from collections import Counter

import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    plot_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import utils


# FUNCTION
def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("KNN RandomUnderSampling confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="BuGn", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def conf_mat_disp(confusion_matrix, disp_labels):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=disp_labels
    )

    disp.plot(cmap="BuGn")


# DATASET
df = utils.load_tracks(
    "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
)

column2drop = [
    ("track", "language_code"),
]

df.drop(column2drop, axis=1, inplace=True)
print(df["album", "type"].unique())

# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "listens"),
    ("album", "type"),
    ("track", "license"),
    ("album", "comments"),
    ("album", "date_created"),
    ("album", "favorites"),
    ("artist", "comments"),
    ("artist", "date_created"),
    ("artist", "favorites"),
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

print(df.info())

# Create KNN Object.
knn = KNeighborsClassifier(
    n_neighbors=5, p=1
)  # valori migliori dalla gridsearch n = 5, p=1, levarli per avere la standard

x = df.drop(columns=[("album", "type")])
y = df[("album", "type")]

X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)
print(X_train.shape, X_test.shape)

knn.fit(X_train, y_train)

# Apply the KNN on the test set and evaluate the performance
print("Apply the KNN on the test set and evaluate the performance: \n")
Y_pred = knn.predict(X_test)
print("Accuracy %s" % accuracy_score(y_test, Y_pred))
print("F1-score %s" % f1_score(y_test, Y_pred, average=None))
print(classification_report(y_test, Y_pred))


"""EMBALANCE LEARNING"""
# PRINTING PCA FOR COMPARISON

print("Train shape")  # , X_train.shape())
pca = PCA(n_components=4)
pca.fit(X_train)
X_pca = pca.transform(X_train)
X_pca.shape

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_train,
    cmap="Set2",
    edgecolor="k",
    alpha=0.5,
)
plt.title("Standard KNN-PCA")
plt.show()


"""UNDERSAMPLING-RANDOM UNDERSAMPLING"""

print(("\033[1m" "Making Undersampling" "\033[0m"))

rus = RandomUnderSampler(random_state=42, replacement=True)
X_res, y_res = rus.fit_resample(X_train, y_train)
print("Original dataset shape %s" % Counter(y_train))
print("Resampled dataset shape %s" % Counter(y_res))

pca = PCA(n_components=4)
pca.fit(X_train)
X_pca = pca.transform(X_res)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_res, cmap="Set2", edgecolor="k", alpha=0.5)
plt.title("KNN-PCA with RandomUnderSampling")
plt.show()

clf = KNeighborsClassifier(n_neighbors=5, p=1)
clf.fit(X_res, y_res)

# Apply the knn on the training set
print("Apply the KNN-UNDERSAMPLE on the training set: \n")
y_pred = clf.predict(X_train)
print("Accuracy knn-undersample %s" % accuracy_score(y_train, y_pred))
print("F1-score knn-undersample %s" % f1_score(y_train, y_pred, average=None))
print(classification_report(y_train, y_pred))

# Apply the KNN on the test set and evaluate the performance
print("Apply the KNN-UNDERSAMPLE on the test set and evaluate the performance: \n")
y_pred = clf.predict(X_test)
print("Accuracy knn-undersample %s" % accuracy_score(y_test, y_pred))
print("F1-score knn-undersample %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))


draw_confusion_matrix(clf, X_test, y_test)

"""ROC Curve"""

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
plt.title("KNN-Undersampling Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()
