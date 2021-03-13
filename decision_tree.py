from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import utils

tracks = utils.load(Path("data/tracks.csv"), clean=True, dummies=True)


target_class = ("album", "type")

attributes = [col for col in tracks.columns if tracks[col].isnull().sum() == 0]
attributes = [col for col in attributes if col != target_class]
X = tracks[attributes].values
y = tracks[target_class]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

X_train.shape, X_test.shape

clf = DecisionTreeClassifier(
    criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1
)
clf.fit(X_train, y_train)

for col, imp in zip(attributes, clf.feature_importances_):
    print(col, imp)
