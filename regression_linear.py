import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import pretty, print
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

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
print("R2: %.3f" % r2_score(test_y, y_pred))
print("MSE: %.3f" % mean_squared_error(test_y, y_pred))
print("MAE: %.3f" % mean_absolute_error(test_y, y_pred))

# i don't know what the fuck this graph code does - copypaste from Guidotti
"""
x_values = sorted(test_x[("track", "listens")])
y_values = y_pred[np.argsort(test_x[("track", "listens")])]
plt.scatter(test_x, test_y, color="black")
plt.plot(x_values, y_values, color="blue", linewidth=3)
plt.title("Scatterplot with regression line")
plt.show()
"""

# this is better graphing, using seaborn
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
column2drop = []

tracks = utils.load_small_tracks(buckets="continuous")

reg = LinearRegression()
reg.fit(all_dfs["train_x"].values, all_dfs["train_y"])

print(reg)

print("Coefficients: \n", reg.coef_)
print("Intercept: \n", reg.intercept_)


y_pred = reg.predict(all_dfs["test_x"])
print("R2: %.3f" % r2_score(all_dfs["test_y"], y_pred))

print("MSE: %.3f" % mean_squared_error(y_test, y_pred))
print("MAE: %.3f" % mean_absolute_error(y_test, y_pred))
