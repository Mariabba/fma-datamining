from prefixspan import PrefixSpan
from music import MusicDB
import numpy as np
import pandas as pd

musi = MusicDB()

each_genre = pd.DataFrame()
for genre in musi.feat["enc_genre"].unique():
    each_genre = each_genre.append(musi.sax[musi.feat["enc_genre"] == genre].head(1))

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

ps = PrefixSpan(X_seq)
ps.minlen = 2
ps.maxlen = 2

print("ps.frequent")
frequent_patterns = ps.frequent(8)

index = []
for i in range(len(X_seq)):
    for j in range(len(frequent_patterns)):
        #if set(frequent_patterns[j][1]).issubset(X_seq[i]):
        if (all(x in X_seq[i] for x in frequent_patterns[j][1])):
            print("Il pattern: ", frequent_patterns[j][1]  ," è stato trovato nella ts ", i)
            index.append(i)

print(index)


