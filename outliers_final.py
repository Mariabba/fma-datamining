import pandas as pd
from rich import print

import utils

df = utils.load_tracks(buckets="continuous", outliers=False)
df = df[[("track", "interest")]]  # dropped all but the needed columns

# load outlier csv's
df_a = pd.read_csv(
    "strange_results_new/abod.csv", index_col="track_id"
)  # 0 if not outlier, 1 if outlier
df_b = pd.read_csv(
    "strange_results_new/dbscan.csv", index_col="track_id"
)  # -1 if outlier
df_c = pd.read_csv(
    "strange_results_new/KNN.csv", index_col="track_id"
)  # False if not outlier, True if outlier
df_d = pd.read_csv("strange_results_new/forest.csv", index_col="track_id")

# print(df_a.loc[:, "0"] == 0)

# filter dataframes and drop unneeded columns
num_outliers = {}
df_a = df[df_a["0"] == 0]
num_outliers["abod"] = len(df) - len(df_a)

df_b = df[df_b["cluster"] != -1]
num_outliers["dbscan"] = len(df) - len(df_b)

df_c = df[df_c["Outlier"] == False]
num_outliers["knn"] = len(df) - len(df_c)

df_d = df[df_d["outliers"] == 1]
num_outliers["forest"] = len(df) - len(df_d)

print(num_outliers)

variance_0 = round(df[("track", "interest")].var() / 1000, 2)  # original variance
# calculate new variances
variance = {}
variance["abod"] = round(df_a[("track", "interest")].var() / 1000, 2)
variance["dbscan"] = round(df_b[("track", "interest")].var() / 1000, 2)
variance["knn"] = round(df_c[("track", "interest")].var() / 1000, 2)
variance["forest"] = round(df_d[("track", "interest")].var() / 1000, 2)

delta_var = {}
delta_var["abod"] = variance_0 - variance["abod"]
delta_var["dbscan"] = variance_0 - variance["dbscan"]
delta_var["knn"] = variance_0 - variance["knn"]
delta_var["forest"] = variance_0 - variance["forest"]

print(f"Original variance: {variance_0}")
results = pd.Series(variance)
results.name = "new variance"
results = pd.DataFrame(results)

delta_var = pd.Series(delta_var)
results["var gain"] = round(delta_var, 2)

num_outliers = pd.Series(num_outliers)
results["# outliers"] = num_outliers

results["unit var gain"] = round(results["var gain"] / results["# outliers"], 2)

print(results.sort_values("unit var gain", ascending=False))
