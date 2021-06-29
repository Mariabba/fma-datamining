# FMA (Free Music Archive) Dataset: an exploration

A project done as students of the course *"Data Mining: Advanced topics and applications"* at [University of Pisa](https://github.com/Unipisa), department of Computer Science.

To read the instructions given to us, we refer you to the [guidelines to the project](http://didawiki.di.unipi.it/doku.php/dm/start#exam_dm_part_ii_dma).

Our gratitude goes to the many people who have explored this dataset before us and to those who created it which are *Defferrard Michael*, *Benzi Kirell*, *Vandergheynst Pierre*, *Bresson Xavier*. We use the music and metadata under MIT and CC BY 4.0 licenses respectively. Please refer to the [dataset's repository](https://github.com/mdeff/fma) for more information.

This project, its code and its report are authored by [Marianna Abbattista](https://github.com/Mariabba), [Fabio Michele Russo](https://github.com/gatto) and [Saverio Telera](https://github.com/Telera), students of Data Science.

## Usage of our interfaces

There are two interfaces to the data: `music.py` (for working with time series of waveforms) and `utils.py` (for working with song metadata). music.py defines a class, **MusicDB()**, while utils.py defines several functions. We use these to ensure smooth/efficient loading and consistent preprocessing. If you want to **explore the data yourself** using our loading and preprocessing techniques, all you need to use are either music.py or utils.py by using the objects or functions described here.

## music.py

```python
from music import MusicDB
musi = MusicDB()
print(musi.df.info())
```

`music.py` provides three objects: `df`, `feat`, `sax`, all contained in the class `MusicDB()`.

### MusicDB.df

A **pandas.DataFrame** with the number of features determined by our *main song* (which is *data/music/000/000002.mp3*) and with as many rows as songs in the small dataset (8000) minus 3 songs dropped for performance reasons. Therefore, the final **shape** of this is **(7997, 2699)**.

### MusicDB.feat

A **dataframe** of metadata on each time series. Holds only genre (textual form) and encoded genre (numeric integer form), but any other interesting metadata on our time series can be added as a column.

### MusicDB.sax

A **dataframe** of **SAX**-ed time series with our best parameters of **130 segments**, **20 symbols**.

## utils.py

```python
import utils
```

`music.py` provides three functions: `load_tracks_xyz()`, `load_tracks()`, `load_small_tracks()`.

### utils.load_tracks_xyz() -> dict

```python
- filepath="data/tracks.csv",
- splits=3,
- buckets="basic",
- dummies=True,
- fill=True,
- outliers=True,
- extractclass=None,
- small=False
```

Parameter usage is the same as `utils.load_tracks()`, below, except for these differences:

#### with splits=3 (default)

**Returns** a dict of 3 **pd.Dataframe** from tracks.csv. The dataframes are contained in a dict for which the keys are _"train"_, _"vali"_, _"test"_.

If **extractclass**=*some_column* this function **returns** a **dict** of 6 items with keys: [*"train_x"*, *"train_y"*, *"vali_x"*, *"vali_y"*, *"test_x"*, *"test_y"*].

Each of the three **_x** versions are type **pd.Dataframe** and contain all the attributes except *some_column*. Each of the three **_y** versions are **pd.Series** and contain just *some_column*. The correct row indexes are retained in all.

#### with splits=2

**Returns** dict of 2 **pd.Dataframe** from tracks.csv: [_"train"_, _"test"_]

If **extractclass**=*some_column* **returns** a **dict** with keys: [*"train_x"*, *"train_y"*, *"test_x"*, *"test_y"*].

#### small

Same usage as above, but returns only the "small" dataset with 10 features + *(album, type)*.

#### Examples

##### Load the three dataframes

```python
import utils
dfs = utils.load_tracks_xyz()

print(dfs['train'].info())
print(dfs['vali'].info())
print(dfs['test'].info())
```

##### You can reassign

```python
train = dfs['train']
print(train.info())
```

##### If you want to extract a class

```python
dfs = utils.load_tracks_xyz(extractclass=("track", "listens"))

print(dfs['train_x'].info())
print(dfs['train_y'].describe())  # Series, contains only ("track", "listens")
```

### utils.load_tracks() -> pd.Dataframe

```python
- filepath="data/tracks.csv",
- buckets="basic",
- dummies=True,
- fill=True,
- outliers=True
```

- filepath should only be changed when you put your files inside of subfolders
- buckets (_basic_, _continuous_, _discrete_): basic is discretizations we use for everything, discrete only for methods who prefer discrete attributes (like decision tree), continuous only for methods who prefer continuous attributes (like knn)
- dummies: Makes dummies out of columns with coverage <10% and a few more special cases hard coded in `utils.dummy_maker()`
- fill: will fill missing values but so far it only deletes rows that contain outliers
- outliers: **if true must be used with fill=_True_** and removes outliers determined by `abod1072.csv`.

**Returns** a single pd.Dataframe.

### utils.load_small_tracks() -> pd.Dataframe

```python
- filepath="data/tracks.csv",
- buckets="basic",
- dummies=True,
- fill=True,
- outliers=True,
```

Same exact usage as `utils.load_tracks()`, but returns only the *10 features* + *(album, type)* we selected.

## Filenaming guidelines

| Module | Filename |
|:--|:--|
1 Starting classification | basic_***method***_***type***.py
1 Anomaly detection | outliers_***method***_***type***.py
1 Imbalanced learning | imbalance_***method***_***type***.py
2 Advanced Classification | advcl_***method***_***type***.py
2 Regression | regression_***method***_***type***.py
3 Time Series | ts_***method***_***type***.py

(***_type*** is optional for all files)

Example:

- imbalance_KNN_over.py
- advcl_SVM_linear.py
- regression_linear.py

## Style guidelines for report

*corsivo* solo nomi colonne

**grassetto** altre cose da rendere evidenti, NON nomi colonne

++commenti per noi tre racchiusi fra due più++

### colonne

- *album* solo corsivo, senza apici ’ o virgolette “
- *(artist, album)* colonna con doppio index, sempre corsivo, racchiusa da parentesi

### parametri

creterion=**gini** grassetto senza virgolette
oppure discorsivamente: for this we used **gini** and …

### Tre puntini di sospensione

usare questo: … che è diverso dai tre singoli punti ...
