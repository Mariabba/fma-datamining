import pandas as pd
from rich import pretty, print
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import utils

pretty.install()

column2drop = [
    ("album", "title"),
    ("album", "type"),
    ("artist", "name"),
    ("track", "title"),
    ("album", "tags"),
    ("artist", "tags"),
    ("track", "language_code"),
    ("track", "license"),
    ("track", "number"),
    ("track", "tags"),
    ("track", "genres"),  # todo da trattare se si vuole inserire solo lei
    ("track", "genres_all"),
]

all_dfs = utils.load_tracks_xyz(buckets="continuous", extractclass=("track", "listens"))

for df in all_dfs:
    try:
        all_dfs[df] = all_dfs[df].drop(column2drop, axis=1)
    except ValueError:
        pass


reg = LinearRegression()
reg.fit(all_dfs["train_x"].values, all_dfs["train_y"])

print(reg)

print("Coefficients: \n", reg.coef_)
print("Intercept: \n", reg.intercept_)


y_pred = reg.predict(all_dfs["test_x"])
print("R2: %.3f" % r2_score(all_dfs["test_y"], y_pred))

exit()

print("MSE: %.3f" % mean_squared_error(y_test, y_pred))
print("MAE: %.3f" % mean_absolute_error(y_test, y_pred))
