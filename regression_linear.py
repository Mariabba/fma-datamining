import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich import pretty, print
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import utils

pretty.install()
console = Console()
# outputs in table format
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Method", style="green")
# table.add_column("Coefficients")
table.add_column("R^2", justify="right")
table.add_column("MSE", justify="right")
table.add_column("MAE", justify="right")

# Univariate
train_x, train_y, test_x, test_y = utils.load_tracks_xyz(
    buckets="continuous", extractclass=("track", "favorites"), splits=2
).values()

train_x = train_x[[("track", "listens")]]
test_x = test_x[[("track", "listens")]]

reg = LinearRegression()
reg.fit(train_x, train_y)

print("[magenta]Univariate[/magenta]")
print("Intercept:\t", reg.intercept_)
print("Coefficients:\t", reg.coef_[0])
y_pred = reg.predict(test_x)

table.add_row(
    "Univariate linear",
    f"{r2_score(test_y.astype(float), y_pred):.3f}",
    f"{mean_squared_error(test_y.astype(float), y_pred):.3f}",
    f"{mean_absolute_error(test_y.astype(float), y_pred):.3f}",
)

# Graphing
train_graphing = train_x.join(train_y).astype(float)
train_graphing.columns = train_graphing.columns.get_level_values(1)
sns.set_theme(style="darkgrid")
g = sns.jointplot(
    x="listens",
    y="favorites",
    data=train_graphing,
    kind="reg",
    truncate=False,
    color="m",
    ci=None,
    height=7,
    scatter_kws={"s": 20},
)
plt.show()


# Lasso
reg = Lasso()
reg.fit(train_x, train_y)
print("[magenta]Lasso[/magenta]")
print("Intercept:\t", reg.intercept_)
print("Coefficients:\t", reg.coef_[0])
y_pred = reg.predict(test_x)
table.add_row(
    "Lasso",
    f"{r2_score(test_y.astype(float), y_pred):.3f}",
    f"{mean_squared_error(test_y.astype(float), y_pred):.3f}",
    f"{mean_absolute_error(test_y.astype(float), y_pred):.3f}",
)


# Ridge
reg = Ridge()
reg.fit(train_x, train_y)
print("[magenta]Ridge[/magenta]")
print("Intercept:\t", reg.intercept_)
print("Coefficients:\t", reg.coef_[0])
y_pred = reg.predict(test_x)
table.add_row(
    "Ridge",
    f"{r2_score(test_y.astype(float), y_pred):.3f}",
    f"{mean_squared_error(test_y.astype(float), y_pred):.3f}",
    f"{mean_absolute_error(test_y.astype(float), y_pred):.3f}",
)


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

print("[magenta]Multivariate[/magenta]")
print("Intercept: ", reg.intercept_)
print("Coefficients: \n", reg.coef_)

y_pred = reg.predict(test_x)

table.add_row(
    "Multivariate linear",
    f"{r2_score(test_y, y_pred):.3f}",
    f"{mean_squared_error(test_y, y_pred):.3f}",
    f"{mean_absolute_error(test_y, y_pred):.3f}",
)

console.print(table)
console.print("[red underline]MSE[/red underline]: Mean squared error")
console.print("[red underline]MAE[/red underline]: Mean absolute error")
