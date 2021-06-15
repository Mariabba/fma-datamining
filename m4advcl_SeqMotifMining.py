import numpy as np
import pandas as pd
from prefixspan import PrefixSpan
from rich import print

from music import MusicDB

musi = MusicDB()

"""each_genre = pd.DataFrame()
for genre in musi.feat["enc_genre"].unique():
    each_genre = each_genre.append(musi.sax[musi.feat["enc_genre"] == genre].head(1))
"""
each_genre = musi.sax
db = each_genre.values

map_symbols = {k: v for v, k in enumerate(np.unique(db.ravel()))}

seq = np.array([map_symbols[v] for v in db.ravel()])

X_seq = list()
for x in db:
    X_seq.append([map_symbols.get(v, -1) for v in x.ravel()])

print(f"sequence 0 is {len(X_seq[0])} long and is {X_seq[0]}")

ps = PrefixSpan(X_seq)
ps.minlen = 2
ps.maxlen = 2

frequent_patterns = [
    (7758, [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
    (7757, [10, 9, 10, 10, 10, 9, 9, 9, 9, 9]),
    (7757, [9, 9, 9, 9, 9, 9, 10, 9, 10, 10]),
    (7755, [9, 10, 10, 10, 9, 9, 9, 9, 9, 9]),
    (7755, [9, 9, 10, 9, 9, 9, 9, 9, 9, 9]),
]


def search_for_motif(seq, pattern):
    internal_i, n, m = -1, len(seq), len(pattern)
    try:
        while True:
            internal_i = seq.index(pattern[0], internal_i + 1, n - m + 1)
            if pattern == seq[internal_i : internal_i + m]:
                return True
    except ValueError:
        return False


index = {}
for j, (support, pattern) in enumerate(frequent_patterns):
    for i, seq in enumerate(X_seq):
        if search_for_motif(seq, pattern):
            try:
                index[j].append(i)
            except KeyError:
                index[j] = [i]

for pattern in index:
    print(f"For motif {pattern} there are {len(index[pattern])} TS that contain it.")

print(index)
