import attr
from numpy import mask_indices
import pandas as pd
import librosa
from pathlib import Path
import matplotlib.pyplot as plt


@attr.s
class MusicDB(object):
    df = attr.ib()

    @df.default
    def _dataframe_default(self):
        self.pick = self._dataframe_pickleload()
        if self.pick:
            return self.pick
        # if not, populate
        return self._dataframe_populate()

    # start of private methods
    def _dataframe_pickleload(self):
        path_to_pickle = Path("data/picks/small.pkl")
        try:
            self.pipi = {}
            self.pipi["df"] = pd.read_pickle(path_to_pickle)
            self.pipi["status"] = True
        except FileNotFoundError:
            return False
        return self.pipi

    def _dataframe_populate(self):
        # extract using librosa from tracks
        self.df = pd.DataFrame(columns=list(range(30)))
        self.df = self.df.append([["miao" for _ in range(30)]], ignore_index=True)

        try:
            y, sr = librosa.load("data/000002.mp3", sr=None)
        except RuntimeError:  # this for Marianna b/c computer runs on hamsters
            my_path = Path.cwd() / "data/000002.mp3"
            y, sr = librosa.load(str(my_path), sr=None)

        miao = librosa.resample(y, sr, 90)
        print(y, sr)
        print(len(y))
        print(miao, len(miao))
        plt.plot(miao)
        plt.show()
        # -> 🐱 remember to save df
        return self.df


if __name__ == "__main__":
    music = MusicDB()
    # some printing just to understand how this works
    print(music.df.info(), music.df.columns)
    print(music.df.head())
