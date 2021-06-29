import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    plot_confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def draw_confusion_matrix(Clf, X, y):
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Rules Based Confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(Clf, X, y, cmap="Reds", normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


prediction_train = pd.read_csv("RuleBasedResults_train_10.csv")
prediction_test = pd.read_csv("RuleBasedResults_test_10.csv")
y_pred_train = []
y_pred_test = []

for index, row in prediction_train.iterrows():
    if row["SingleTracks_y_pred_train"] == True:
        y_pred_train.append("Single Tracks")
    elif row["LivePerformance_y_pred_train"] == True:
        y_pred_train.append("Live Performance")
    elif row["RadioProgram_y_pred_train"] == True:
        y_pred_train.append("Radio Program")
    else:
        # default case
        y_pred_train.append("Album")

for index, row in prediction_test.iterrows():
    if row["SingleTracks_y_pred_test"] == True:
        y_pred_test.append("Single Tracks")
    elif row["LivePerformance_y_pred_test"] == True:
        y_pred_test.append("Live Performance")
    elif row["RadioProgram_y_pred_test"] == True:
        y_pred_test.append("Radio Program")
    else:
        # default case
        y_pred_test.append("Album")

prediction_train["y_pred"] = y_pred_train
prediction_test["y_pred"] = y_pred_test

# Apply the decision tree on the training set
print("Apply the decision tree on the training set: \n")
y_pred = prediction_train["y_pred"]
y_train = prediction_train["y_train"]
print("Accuracy %s" % accuracy_score(y_train, y_pred))
print("F1-score %s" % f1_score(y_train, y_pred, average=None))
print(classification_report(y_train, y_pred))
confusion_matrix(y_train, y_pred)

# Apply the decision tree on the test set and evaluate the performance
print("Apply the decision tree on the test set and evaluate the performance: \n")
y_pred = prediction_test["y_pred"]
y_test = prediction_test["y_test"]
print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
# draw_confusion_matrix(ripper_clf, X_test, y_test)

# ROC Curve
from sklearn.preprocessing import LabelBinarizer

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
print(roc_auc)

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
plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)
plt.tick_params(axis="both", which="major", labelsize=22)
plt.legend(loc="lower right", fontsize=14, frameon=False)
plt.show()
