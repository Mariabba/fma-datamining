"""libraries"""
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from music import MusicDB
import scipy.stats as stats

from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

"""
FILE 4  -CLASSIFICAZIONE BINARIA GENERE BASE (SCEGLIERE LA CLASSE TARGET)

In questo file vi sono 2 tipologie di classifcazione base per le TS:

2- Distance Based Class-KNN
3- Random Forest
"""
