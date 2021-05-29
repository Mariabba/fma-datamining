import pandas as pd

import utils

df = utils.load_tracks(buckets="continuous", outliers=False)
df  # dropped all but the columns
variance_0  # original variance

# load outlier csv's
df_a = pd.read_csv("strange_results/new/abod.csv")  # 0 if not outlier, 1 if outlier
df_b = "dbscan outliers"
df_c = "knn outliers"
df_d = "forest outliers"

# filter dataframes and drop unneeded columns
df_a = df[df_a["0"] == 0]
df_b
df_c
df_d

# calculate new variances
variance_1
variance_2
variance_3
variance_4

delta_var = {}
delta_var["abod"]
delta_var["dbscan"]
delta_var["knn"]
delta_var["forest"]

results = pd.Series(delta_var)

print(results)
