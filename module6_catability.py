import lime
import lime.lime_tabular
import numpy as np
from lime import lime_tabular
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

import utils

df = utils.load_small_tracks(buckets="discrete")
# df = df.head(100)
# CAMBIO ALBUM TYPE IN BINARIA
# print("prima", df["album", "type"].unique())
# df["album", "type"] = df["album", "type"].replace(
#    ["Single Tracks", "Live Performance", "Radio Program"],
#    ["NotAlbum", "NotAlbum", "NotAlbum"],
# )
# print("dopo", df["album", "type"].unique())


label_encoders = dict()
column2encode = [
    ("track", "duration"),
    ("track", "interest"),
    ("track", "listens"),
    # ("album", "type"),
]
for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
df.info()
# print(df[df["album", "type"] == "NotAlbum"].head())
class_name = ("album", "type")

df["index-cat"] = df.index  # add index as last column
attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=17,
    min_samples_split=3,
    min_samples_leaf=3,
    max_features="auto",
    random_state=10,
    class_weight="balanced",
)


clf.fit(X_train, y_train)

# text_representation = tree.export_text(clf)
# print(text_representation)

score = clf.score(X_test, y_test)


def bb_predict(X):
    return clf.predict(X)


def bb_predict_proba(X):
    return clf.predict_proba(X)


y_pred = bb_predict(X_test)

print("Accuracy %.3f" % accuracy_score(y_test, y_pred))
# print("F1-measure %.3f" % f1_score(y_test, y_pred))

"""LIME
lime_explainer = LimeTabularExplainer(
    X_test,
    feature_names=df.columns,
    class_names=[str(v) for v in df.values],
    discretize_continuous=False,
)

exp = lime_explainer.explain_instance(X_test, bb_predict_proba)

print(exp.local_exp)

# print(exp.show_in_notebook())
"""

class_names = ["Album", "Single Tracks", "Live Performance", "Radio Program"]

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=df.columns,
    class_names=class_names,
    mode="classification",
)
X_test = pd.DataFrame(X_test[:, :-1], columns=df.columns[1:-1], index=X_test[:, -1])
print(X_test.info())

for one_class in class_names:
    records = X_test[X_test[("artist", "website")] == 1].tail(5)
    print(records)
    for record in records:
        print(record)
        i2e = record.index
        x = record.values
        exp = explainer.explain_instance(data_row=x, predict_fn=clf.predict_proba)
        print(exp.local_exp)
        exp.save_to_file("porco.html")
# bb_outcome = bb_predict(x.reshape(1, -1))[0]
# bb_outcome_str = df.values[bb_outcome]

# print("bb(x) = { %s }" % bb_outcome_str)
