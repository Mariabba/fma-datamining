from lime import lime_tabular
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import lime
import lime.lime_tabular
import utils
import seaborn as sns
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer

df = utils.load_small_tracks(buckets="discrete")
df = df.head(100)
# CAMBIO ALBUM TYPE IN BINARIA
print("prima", df["album", "type"].unique())
df["album", "type"] = df["album", "type"].replace(
    ["Single Tracks", "Live Performance", "Radio Program"],
    ["NotAlbum", "NotAlbum", "NotAlbum"],
)
print("dopo", df["album", "type"].unique())


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


class_name = ("album", "type")

attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
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


import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=df.columns,
    class_names=["album", "not album"],
    mode="classification",
)
i2e = 2
x = X_test[i2e]
# bb_outcome = bb_predict(x.reshape(1, -1))[0]
# bb_outcome_str = df.values[bb_outcome]

# print("bb(x) = { %s }" % bb_outcome_str)
print("")

exp = explainer.explain_instance(data_row=x, predict_fn=clf.predict_proba)

print(exp.show_in_notebook(show_table=True))
print(exp.local_exp)
exp.save_to_file("C:/Users/jigok/OneDrive/Desktop/porco.html")
