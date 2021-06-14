from music import MusicDB

musi = MusicDB()

records5 = musi.df.iloc[0:5, :]

records5.to_csv("first5_ts.csv")
