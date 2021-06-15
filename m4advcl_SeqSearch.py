from prefixspan import PrefixSpan
from music import MusicDB
import numpy as np
import pandas as pd
from rich import print

musi = MusicDB()
"""
each_genre = pd.DataFrame()
for genre in musi.feat["enc_genre"].unique():
    each_genre = each_genre.append(musi.sax[musi.feat["enc_genre"] == genre].head(100))
"""
each_genre = musi.sax
print(each_genre)
print(each_genre.info())
db = each_genre.values

map_symbols = {k: v for v, k in enumerate(np.unique(db.ravel()))}
print(f"Symbols found: {map_symbols}")

seq = np.array([map_symbols[v] for v in db.ravel()])

X_seq = list()
for x in db:
    X_seq.append([map_symbols.get(v, -1) for v in x.ravel()])

ps = PrefixSpan(X_seq)
ps.minlen = 14
ps.maxlen = 14
support = 5000

hello = ps.frequent(support)
print(
    f"With minlen={ps.minlen}, maxlen={ps.maxlen}, support={support}, I found {len(hello)} frequent patterns. Here you go:"
)
print(sorted(hello, reverse=True))

"""
print("ps.topk")
print(ps.topk(10))
"""
