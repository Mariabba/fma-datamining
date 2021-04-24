import attr
import pandas as pd
import librosa


@attr.s
class MusicDB(object):
    _dataframe = attr.ib()

    @_dataframe.default
    def _dataframe_default(self):
        # check if we have the pickle
        # if self._dataframe_picklecheck: etc
        # return self._dataframe_pickeload

        # if not, populate
        return self._dataframe_populate()

    def _dataframe_picklecheck(self):
        # check if we have pickle

        pass

    def _dataframe_populate(self):
        # extract using librosa from tracks
        self.df = pd.DataFrame(columns=list(range(30)))
        self.df = self.df.append([["miao" for _ in range(30)]], ignore_index=True)

        # some printing just to understand how this works
        print(self.df.info(), self.df.columns)
        print(self.df.head())
        return self.df


music = MusicDB()
