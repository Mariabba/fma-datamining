import attr
from numpy import blackman, mask_indices
import pandas as pd
import librosa
from pathlib import Path
import matplotlib.pyplot as plt


@attr.s
class MusicDB(object):
    df = attr.ib()

    @df.default
    def _dataframe_default(self):
        pick = self._dataframe_pickleload()
        if pick:
            return pick
        # if not, populate
        return self._dataframe_populate()

    # start of private methods
    def _dataframe_pickleload(self):
        path_to_pickle = Path("data/picks/small.pkl")
        try:
            pipi = pd.read_pickle(path_to_pickle)
        except FileNotFoundError:
            return False
        return pipi

    def _dataframe_populate(self):
        # estabilish number of features
        y, sr = librosa.load("data/music/000002.mp3", sr=None)
        miao = librosa.resample(y, sr, 90)
        miao = len(miao)

        # make df
        dfm = pd.DataFrame(columns=list(range(miao)))

        # populate collection of paths of mp3s
        p = Path("data/music").glob("**/*")
        tracks = [x for x in p if x.is_file()]

        # populate df
        for track in tracks:
            # extract waveform and convert
            y, _ = librosa.load(track, sr=None)
            miao = librosa.resample(y, sr, 90)

            # fix the index
            miao = pd.Series(data=miao)
            miao.name = int(track.stem)

            # append to dfm
            dfm = dfm.append(miao)

        # -> 🐱 remember to save df
        return dfm


if __name__ == "__main__":
    music = MusicDB()
    # some printing just to understand how this works
    print(music.df.info())
    print(music.df.head())
