import numpy as np
from rich import print

from music import MusicDB

musi = MusicDB()

each_genre = musi.sax
db = each_genre.values

map_symbols = {k: v for v, k in enumerate(np.unique(db.ravel()))}

seq = np.array([map_symbols[v] for v in db.ravel()])

X_seq = list()
for x in db:
    X_seq.append([map_symbols.get(v, -1) for v in x.ravel()])

frequent_patterns = [
    (7758, [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
    (7757, [10, 9, 10, 10, 10, 9, 9, 9, 9, 9]),
    (7757, [9, 9, 9, 9, 9, 9, 10, 9, 10, 10]),
    (7755, [9, 10, 10, 10, 9, 9, 9, 9, 9, 9]),
    (7755, [9, 9, 10, 9, 9, 9, 9, 9, 9, 9]),
    (7754, [9, 10, 10, 9, 9, 9, 9, 9, 9, 9]),
    (7754, [9, 9, 9, 10, 9, 10, 10, 9, 9, 9]),
    (7754, [9, 9, 9, 9, 9, 10, 9, 10, 10, 10]),
    (7753, [9, 10, 9, 10, 10, 9, 9, 9, 9, 9]),
    (7753, [9, 10, 9, 9, 9, 9, 9, 9, 9, 9]),
    (7753, [9, 9, 10, 10, 10, 9, 9, 9, 9, 9]),
    (7753, [9, 9, 10, 10, 9, 9, 9, 9, 9, 9]),
    (7753, [9, 9, 9, 10, 9, 9, 9, 9, 9, 9]),
    (7753, [9, 9, 9, 9, 9, 9, 9, 10, 10, 10]),
]

index = {}
for j, (support, pattern) in enumerate(frequent_patterns):
    for i, ts in enumerate(X_seq):
        if all(x in ts for x in pattern):
            try:
                index[j].append(i)
            except KeyError:
                index[j] = [i]

print(index)

for pat in index:
    print(len(index[pat]))
