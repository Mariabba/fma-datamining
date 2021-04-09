from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

import utils

dfs = utils.load_tracks_xyz(extractclass=("track", "listens"))


scaler = StandardScaler()
scaler.fit(dfs["train_x"])
train_x = scaler.transform(dfs["train_x"])
test_x = scaler.transform(dfs["test_x"])
dfs["train_y"] = pd.DataFrame(dfs["train_y"])
train_y = scaler.transform(dfs["train_y"])

clf = MLPClassifier(random_state=0)

clf.fit(train_x, train_y)

exit()

y_pred = clf.predict(test_x)

print("Accuracy %s" % accuracy_score(y_test, y_pred))
print("F1-score %s" % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
