from prefixspan import PrefixSpan
from music import MusicDB
import numpy as np


musi = MusicDB()


print(musi.sax.info())
db = musi.sax.head(5).values

map_symbols = {k: v for v, k in enumerate(np.unique(db.ravel()))}
print(map_symbols)

seq = np.array([map_symbols[v] for v in db.ravel()])
print(seq)
X_seq = list()
for x in db:
    X_seq.append([map_symbols.get(v, -1) for v in x.ravel()])


ps = PrefixSpan(X_seq)

print(ps.frequent(3))

print(ps.topk(10))

"""
# minsup standard = 2
p.setminsup(5)
p.setlen(4)
p.run()
# print(p.out())
"""