import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder

import utils

# FUNCTION


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("KNN confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Purples", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def conf_mat_disp(confusion_matrix, disp_labels):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=disp_labels
    )

    disp.plot(cmap="Purples")


# DATASET
# DATASET
# df = utils.load_tracks(
#   "data/tracks.csv", dummies=True, buckets="continuous", fill=True, outliers=True
# )

dfs = utils.load_tracks_xyz(
    buckets="continuous", splits=2, extractclass=("album", "type")
)

column2drop = [
    ("track", "language_code"),
]

dfs["train_x"].drop(column2drop, axis=1, inplace=True)
dfs["test_x"].drop(column2drop, axis=1, inplace=True)
# print(["album", "type"].unique())

# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "listens"),
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
    dfs["train_x"][col] = le.fit_transform(dfs["train_x"][col])
    dfs["test_x"][col] = le.fit_transform(dfs["test_x"][col])
    label_encoders[col] = le

le1 = LabelEncoder()
dfs["train_y"] = le1.fit_transform(dfs["train_y"])
dfs["test_y"] = le1.fit_transform(dfs["test_y"])
label_encoders[("album", "type")] = le1
"""
# Create KNN Object.
knn = KNeighborsClassifier(
    n_neighbors=2, p=1
)  # valori migliori dalla gridsearch n = 2, p=1, levarli per avere la standard
# Create x and y variables.
x = df.drop(columns=[("album", "type")])
y = df[("album", "type")]
# Split data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)
print(X_train.shape, X_test.shape)
# Training the model.
knn.fit(X_train, y_train)

# Apply the knn on the training set
print("Apply the KNN on the training set: \n")
y_pred = knn.predict(X_train)
print("Accuracy %s" % accuracy_score(y_train, y_pred))
print("F1-score %s" % f1_score(y_train, y_pred, average=None))
print(classification_report(y_train, y_pred))

# Apply the KNN on the test set and evaluate the performance
print("Apply the KNN on the test set and evaluate the performance: \n")
Y_pred = knn.predict(X_test)
print("Accuracy %s" % accuracy_score(y_test, Y_pred))
print("F1-score %s" % f1_score(y_test, Y_pred, average=None))
print(classification_report(y_test, Y_pred))
draw_confusion_matrix(knn, X_test, y_test)


#""" """ROC Curve""" """

lb = LabelBinarizer()
lb.fit(y_test)
lb.classes_.tolist()

fpr = dict()
tpr = dict()
roc_auc = dict()
by_test = lb.transform(y_test)
by_pred = lb.transform(Y_pred)
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
plt.title("KNN  Roc-Curve")
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="lower right", fontsize=7, frameon=False)
plt.show()
"""
