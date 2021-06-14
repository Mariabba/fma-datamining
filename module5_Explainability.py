import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import utils

df = utils.load_small_tracks(buckets="discrete")

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

print(df["album", "type"].unique())
class_name = ("album", "type")

attributes = [col for col in df.columns if col != class_name]
X = df[attributes].values
y = df[class_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

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
score = clf.score(X_test, y_test)


def bb_predict(X):
    return clf.predict(X)


y_pred = bb_predict(X_test)

print("Accuracy %.3f" % accuracy_score(y_test, y_pred))
# print("F1-measure %.3f" % f1_score(y_test, y_pred))


from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=df[attributes].columns,
    class_names=["Album", "Single Tracks", "Live Performance", "Radio Program"],
    mode="classification",
)
i2e = 10993
# i2e = np.random.randint(0, X_test.shape[0])
x = X_test[i2e]

print("")

exp = explainer.explain_instance(
    data_row=x, predict_fn=clf.predict_proba, num_features=4, top_labels=4
)

print("Document id: %d" % i2e)
print("miao")

print(exp.local_exp)

exp.save_to_file("porco.html")

"""
Indici da tenere

Stranamente quasi bilanciata:  Document id: 10993

Live Performance : Document id: 16178 
Radio Programm : Document id: 15019
Album : Document id: 11710
Single Tracks : Document id: 9326(73%) or 10236(77%)

"""
