from prefixspan import PrefixSpan
from music import MusicDB
import numpy as np
import pandas as pd

musi = MusicDB()

each_genre = pd.DataFrame()
for genre in musi.feat["enc_genre"].unique():
    each_genre = each_genre.append(musi.sax[musi.feat["enc_genre"] == genre].head(100))

print(each_genre)
print(each_genre.info)
db = each_genre.values

map_symbols = {k: v for v, k in enumerate(np.unique(db.ravel()))}
print(map_symbols)

seq = np.array([map_symbols[v] for v in db.ravel()])

X_seq = list()
for x in db:
    X_seq.append([map_symbols.get(v, -1) for v in x.ravel()])

#print(X_seq)
"""
ps = PrefixSpan(X_seq)
ps.minlen = 4
ps.maxlen = 50

print("ps.frequent")
print(ps.frequent(1000))
"""
"""
print("ps.topk")
print(ps.topk(10))
"""
"""
# minsup standard = 2
p.setminsup(5)
p.setlen(4)
p.run()
# print(p.out())
"""

