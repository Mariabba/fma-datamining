import matplotlib.pyplot as plt

# sklearn
from sklearn import metrics
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

import utils
from pathlib import Path

# ==========FUNCTION==========================
from Classification.decision_tree import X_train, y_train, X_test, y_test


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="OrRd", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def conf_mat_disp(confusion_matrix, disp_labels):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=disp_labels
    )

    disp.plot(cmap="OrRd")


def draw_precision_recall_curve(Y_test, Y_pred):
    fig, ax = plt.subplots()

    pr_ap = average_precision_score(Y_test, Y_pred, average=None)
    precision, recall, ap_thresholds = precision_recall_curve(Y_test, Y_pred)

    ax.plot(precision, recall, color="#994D00", label="AP %0.4f" % (pr_ap))
    # ax.plot([0, 1], [no_skill, no_skill], 'r--', label='%0.4f' % no_skill)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.tick_params(axis="both")
    ax.legend(loc="upper right", title="AP", frameon=True)
    ax.set_title("Model Precision-Recall curve")

    fig.tight_layout()
    plt.show()


# =====================DATASET ADJUSTING AND VISUALIZING===============================
df = utils.load("../data/tracks.csv", dummies=True)
column2drop = [
    ("album", "title"),
    ("album", "tags"),  # might be usefull to include them, but how?
    ("album", "id"),
    ("set", "split"),
    ("track", "title"),
    ("artist", "id"),
    ("artist", "name"),
    ("artist", "tags"),  # might be usefull to include them, but how?
    ("track", "tags"),
    ("track", "genres"),
    ("track", "genres_all"),
]
df.drop(column2drop, axis=1, inplace=True)

# feature to reshape
label_encoders = dict()
column2encode = [
    ("album", "comments"),
    ("album", "favorites"),
    ("album", "listens"),
    ("album", "type"),
    ("artist", "comments"),
    ("artist", "favorites"),
    ("track", "duration"),
    ("track", "comments"),
    ("track", "favorites"),
    ("track", "language_code"),
    ("track", "license"),
    ("track", "listens"),
]

for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print(df.info())


# ==== SUDDIVISIONE DATASET======
def tuning_param(df, target1, target2):
    # split dataset train and set
    attributes = [col for col in df.columns if col != (target1, target2)]
    X = df[attributes].values
    y = df[target1, target2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25
    )
    print(X_train.shape, X_test.shape)

    print("Parameter Tuning: \n")


# ===================FEATURE SELECTION============================

features_sel = [col for col in df.columns if col != ("album", "type")]

X_features_sel = df[features_sel].values
y_features_sel = df[("album", "type")]

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ===========MODEL KNN========================


# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = knn.predict(X_test)


# Model Accuracy, how often is the classifier correct?
draw_confusion_matrix
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# confusion matrix
print("\033[1m" "Confusion matrix" "\033[0m")

# plot_confusion_matrix(clf, X_test_normalized, y_test, cmap = 'OrRd')
draw_confusion_matrix(knn, X_test, y_test)

print()


print("\033[1m" "Classification report test" "\033[0m")
print(classification_report(y_test, y_pred))

print()

print("\033[1m" "Metrics" "\033[0m")
print()

print("Accuracy %s" % accuracy_score(y_test, y_pred))

print("F1-score %s" % f1_score(y_test, y_pred, labels=[0, 1], average=None))

print("Precision %s" % precision_score(y_test, y_pred, labels=[0, 1], average=None))

print("Recall %s" % recall_score(y_test, y_pred, labels=[0, 1], average=None))
