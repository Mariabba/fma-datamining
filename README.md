# Guidelines

## Module 1 - Introduction, Imbalanced Learning and Anomaly Detection

**1-Explore and prepare the dataset.**  You are allowed to take inspiration from the associated GitHub repository and figure out your personal research perspective (from choosing a subset of variables to the class to predict…). You are welcome in creating new variables and performing all the pre-processing steps the dataset needs.

**2-Define one or more (simple) classification tasks** and solve it with Decision Tree and KNN. You decide the target variable.

**3-Identify the top 1% outliers:** adopt at least three different methods from different families (e.g., density-based, angle-based… ) and compare the results. Deal with the outliers by removing them from the dataset or by treating the anomalous variables as missing values and employing replacement techniques. In this second case, you should check that the outliers are not outliers anymore. Justify your choices in every step.

**4-Analyze the value distribution of the class to predict with respect to point 2;** if it is unbalanced leave it as it is, otherwise turn the dataset into an imbalanced version (e.g., 96% - 4%, for binary classification). Then solve the classification task using the Decision Tree or the KNN by adopting various techniques of imbalanced learning.

**5-Draw your conclusions** about the techniques adopted in this analysis.

N.B. When “solving the classification task”, remember, (i) to test, when needed, different criteria for the parameter estimation of the algorithms, and (ii) to evaluate the classifiers (e.g., Accuracy, F1, Lift Chart) in order to compare the results obtained with an imbalanced technique against those obtained from using the “original” dataset.

## Module 2 - Advanced Classification Methods

**1-Solve the classification task** defined in Module 1 (or define new ones) with the other classification methods analyzed during the course:*Naive Bayes Classifier, Logistic Regression, Rule-based Classifiers, Support Vector Machines, Neural Networks, Ensemble Methods* and evaluate each classifier with the techniques presented in Module 1 (accuracy, precision, recall, F1-score, ROC curve). Perform hyper-parameter tuning phases and justify your choices.

**2-Besides the numerical evaluation draw your conclusions** about the various classifiers, e.g. for Neural Networks: what are the parameter sets or the convergence criteria which avoid overfitting? For Ensemble classifiers how the number of base models impacts the classification performance? For any classifier which is the minimum amount of data required to guarantee an acceptable level of performance? Is this level the same for any classifier? What is revealing the feature importance of Random Forests?

**3-Select two continuous attributes, define a regression problem** and try to solve it using different techniques reporting various evaluation measures. Plot the two-dimensional dataset. Then generalize to multiple linear regression and observe how the performance varies.

## Module 3 - Time Series Analysis

**1-Select the feature(s) you prefer and use it (them) as a time series.** You can use the temporal information provided by the authors’ datasets, but you are also welcome in exploring the .mp3 files to build your own dataset of time series according to your purposes. You should prepare a dataset on which you can run time series clustering; motif/anomaly discovery and classification.
**2-On the dataset created, compute clustering** based on Euclidean/Manhattan and DTW distances and compare the results. To perform the clustering you can choose among different distance functions and clustering algorithms. Remember that you can reduce the dimensionality through approximation. Analyze the clusters and highlight similarities and differences.
**3-Analyze the dataset for finding motifs and/or anomalies.** Visualize and discuss them and their relationship with other features.
**4-Solve the classification task on the time series dataset(s) and evaluate each result.** In particular, you should use shapelet-based classifiers. Analyze the shapelets retrieved and discuss if there are any similarities/differences with motifs and/or shapelets.





# Usage
## utils.py
import utils

### utils.load_tracks_xyz -> dict
```python
- filepath="data/tracks.csv",
- splits=3,
- buckets="basic",
- dummies=True,
- fill=True,
- outliers=True,
- extractclass=None
```

Parameter usage is the same as `utils.load_tracks()`, below, except for these differences:

#### with splits=3 (default)
**Returns** a dict of 3 **pd.Dataframe** from tracks.csv. The dataframes are contained in a dict for which the keys are _"train"_, _"vali"_, _"test"_.

If **extractclass**=*some_column* this function **returns** a **dict** of 6 items with keys: [*"train_x"*, *"train_y"*, *"vali_x"*, *"vali_y"*, *"test_x"*, *"test_y"*].

Each of the three **_x** versions are type **pd.Dataframe** and contain all the attributes except *some_column*. Each of the three **_y** versions are **pd.Series** and contain just *some_column*. The correct row indexes are retained in all.

#### with splits=2
**Returns** dict of 2 **pd.Dataframe** from tracks.csv: [_"train"_, _"test"_]

If **extractclass**=*some_column* **returns** a **dict** with keys: [*"train_x"*, *"train_y"*, *"test_x"*, *"test_y"*].

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

### utils.load_tracks -> pd.Dataframe
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

# Filenaming guidelines
| Module | Filename |
|:--|:--|
1 Starting classification | starting_***method***_***type***.py
1 Anomaly detection | outliers_***method***_***type***.py
1 Imbalanced learning | imbalance_***method***_***type***.py
2 Advanced Classification | advcl_***method***_***type***.py
2 Regression | regression_***method***_***type***.py

(_***type*** is optional for all files)

Example:
- imbalance_KNN_over.py
- advcl_SVM_linear.py
- regression_linear.py
