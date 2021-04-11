import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import pretty, print
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import utils

pretty.install()

# Univariate
train_x, train_y, vali_x, vali_y, test_x, test_y = utils.load_tracks_xyz(
    buckets="continuous", extractclass=("track", "favorites")
).values()

train_x = train_x[[("track", "listens")]]
test_x = test_x[[("track", "listens")]]
vali_x = vali_x[[("track", "listens")]]

reg = LinearRegression()
reg.fit(train_x, train_y)

print("Coefficients: \n", reg.coef_)
print("Intercept: \n", reg.intercept_)

y_pred = reg.predict(test_x)
print("R2: %.3f" % r2_score(test_y.astype(float), y_pred))
print("MSE: %.3f" % mean_squared_error(test_y.astype(float), y_pred))
print("MAE: %.3f" % mean_absolute_error(test_y.astype(float), y_pred))

# Graphing
train_x = train_x.join(train_y).astype(float)
train_x.columns = train_x.columns.get_level_values(1)
sns.set_theme(style="darkgrid")

# Show the results of a linear regression within each dataset
g = sns.jointplot(
    x="listens",
    y="favorites",
    data=train_x,
    kind="reg",
    truncate=False,
    color="m",
    ci=None,
    height=7,
    scatter_kws={"s": 20},
)
plt.show()

# Multivariate
train_x, train_y, vali_x, vali_y, test_x, test_y = utils.load_tracks_xyz(
    buckets="continuous", extractclass=("track", "favorites")
).values()

label_encoders = dict()
column2encode = [
    ("track", "language_code"),
    ("album", "type"),
    ("track", "license"),
    ("album", "date_created"),
    ("artist", "date_created"),
    ("track", "date_created"),
    ("track", "duration"),
]
for col in column2encode:
    le = LabelEncoder()
    train_x[col] = le.fit_transform(train_x[col])
    test_x[col] = le.fit_transform(test_x[col])
    label_encoders[col] = le

le = LabelEncoder()
train_y = le.fit_transform(train_y)
test_y = le.fit_transform(test_y)
label_encoders[("track", "favorites")] = le

reg = LinearRegression()
reg.fit(train_x, train_y)

print("Coefficients: \n", reg.coef_)
print("Intercept: \n", reg.intercept_)

y_pred = reg.predict(test_x)
print("R2: %.3f" % r2_score(test_y, y_pred))
print("MSE: %.3f" % mean_squared_error(test_y, y_pred))
print("MAE: %.3f" % mean_absolute_error(test_y, y_pred))
