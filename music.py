import multiprocessing
import sys
import warnings
from pathlib import Path

import attr
import librosa
import pandas as pd
from rich.progress import BarColumn, Progress, TimeRemainingColumn

if not sys.warnoptions:
    warnings.simplefilter("ignore")


@attr.s
class MusicDB(object):
    df = attr.ib()

    @df.default
    def _dataframe_default(self):
        pick = self._dataframe_pickleload()
        if type(pick) is not bool:
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
        # estabilish number of features using the main song
        y, sr = librosa.load("data/music/000/000002.mp3", sr=None)
        miao = librosa.resample(y, sr, 90)
        number_of_feat = len(miao)

        # make df
        print(f"Building a dataframe with {number_of_feat} features.")
        dfm = pd.DataFrame(columns=list(range(number_of_feat)))
        num_errors = 0

        # populate collection of paths of mp3s
        p = Path("data/music").glob("**/*.mp3")
        tracks = [x for x in p if x.is_file()]
        print(f"Making a Dataframe of len {len(tracks)}.")

        # make progress reporting
        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "{task.completed} of {task.total}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        )

        # populate df
        with progress:
            task_id = progress.add_task("[cyan]Extracting...", total=len(tracks))
            with multiprocessing.Pool() as pool:
                for row in pool.imap_unordered(self._do_one_song, tracks):
                    if type(row) is not bool:
                        dfm = dfm.append(row)
                    else:
                        num_errors += 1
                    progress.advance(task_id)

        dfm = dfm.sort_index()
        # ensure the shape is the one of the main song
        dfm = dfm.loc[:, : number_of_feat - 1]
        print(f"There were {dfm.shape[0] * dfm.shape[1] - dfm.count().sum()} NaN.")
        print(f"There also were {num_errors} errors.")
        dfm = dfm.fillna(value=0)
        dfm.to_pickle("data/picks/small.pkl")
        return dfm

    def _do_one_song(self, song):
        # extract waveform and convert
        try:
            y, sr = librosa.load(str(song), sr=None)
            miao = librosa.resample(y, sr, 120)
            # fix the index
            miao = pd.Series(data=miao)
            miao.name = int(song.stem)
            return miao
        except:
            return False


if __name__ == "__main__":
    music = MusicDB()
    # some printing just to understand how this works
    print(music.df.info())
    print(music.df.head())
