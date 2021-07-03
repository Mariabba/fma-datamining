#!/usr/bin/env python
# coding: utf-8

# In[46]:


# general libraries
import os
import sys
import math
import statistics
import collections
from gettext import install
import jupyter
import luxwidget
import missingno as msno
import py
from luxwidget import nbextension
from pylab import MaxNLocator
from collections import defaultdict
import seaborn as sns
import ast
from pathlib import Path
# pandas libraries
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd

# visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# numpy libraries
import numpy as np
from numpy import std
from numpy import mean
from numpy import percentile

# scipy libraries
from scipy.stats import pearsonr

# sklearn libraries
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer  # explicitly require this experimental feature
from sklearn.impute import IterativeImputer


# In[47]:


sns.set()


# In[48]:


tracks = pd.read_csv(r'C:\Users\jigok\PycharmProjects\DataMining2-project\data\tracks.csv',low_memory=False,index_col=0, header=[0, 1])

COLUMNS = [('track', 'tags'), ('album', 'tags'),('artist', 'tags'),
           ('track', 'genres'), ('track', 'genres_all')]

for column in COLUMNS:
     tracks[column] = tracks[column].map(ast.literal_eval)
        
COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
           ('album', 'date_created'), ('album', 'date_released'),
           ('artist', 'date_created'), ('artist', 'active_year_begin'),
           ('artist', 'active_year_end')]
for column in COLUMNS:
     tracks[column] = pd.to_datetime(tracks[column])
        
COLUMNS = [('track', 'genre_top'), ('track', 'license'),
           ('album', 'type'), ('album', 'information'),
           ('artist', 'bio')]
for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')


# In[49]:


tracks.info()


# In[50]:


ipd.display(tracks['track'].head())
ipd.display(tracks['album'].head())
ipd.display(tracks['artist'].head())
ipd.display(tracks['set'].head())


# In[51]:


top_data = msno.nullity_filter(tracks, filter='bottom', n= 50)


# # Missing Value detection

# In[52]:


msno.matrix(top_data)


# In[53]:


fig = plt.subplots(figsize=(15, 5))
fig_dims = (1, 1)

ax = plt.subplot2grid(fig_dims, (0, 0))
msno.matrix(tracks, ax=ax,sparkline=False)
plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=5)
plt.plot()


# In[54]:


msno.bar(top_data)


# In[55]:


msno.heatmap(tracks)


# Zero Value Detection

# In[56]:


data_types = tracks.dtypes
for column_name, column_type in data_types.items():
    if column_type == np.int or column_type == np.float:
        zero_values = (tracks[column_name] == 0).sum()
        if zero_values > 0:
            print(column_name, zero_values, sep="\t")


# Gli attributi numerici possono presentare missing value ma non ci sono in questo caso, ottimo

# # Columns Unique Value Lookup

# Elimino le colonne con valori strani tipo (album, tags) prima, senno mi da errore

# In[57]:


tracks.info()


# In[58]:


#del tracks["album", "tags"]
#del tracks["artist", "tags"]
#del tracks["track", "tags"]


# In[59]:


#column_names = tuple(tracks.columns)
#for column_name in column_names:
 #   unique_values = tracks[column_name].unique()
  #  if len(unique_values) >= 10:
   #     print(column_name, "more than 10 %s values" % tracks.dtypes[column_name], sep="\t")
    #else:
     #    print(column_name, unique_values, sep="\t")


# # Correlation

# In[60]:


def draw_sns_heatmap(correlation_matrix, h_neg, h_pos, title, pngfile):
    """
    Function which draws a seaborn' heatmap based on the correlation matrix passed by argument
    """
    # generate a mask for the upper triangle
    mask = np.zeros_like(correlation_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(40, 10))

    # generate a custom diverging colormap
    cmap = sns.diverging_palette(h_neg, h_pos, as_cmap=True)
    
    # draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)
    plt.xticks(fontsize=10)


# In[61]:


# compute the correlation matrix, excluding NA/null values
correlation_matrix = tracks.corr("pearson")  # pearson correlation
# draw heatmap
draw_sns_heatmap(correlation_matrix, 0, 250, "Pearson correlation matrix", "pearson_correlation_matrix.png")


# # First Look Distribution

# In[62]:


fig = plt.subplots(figsize=(20, 15))
fig_dims =(3,1)

ax = plt.subplot2grid(fig_dims, (0, 0))
sns.countplot(x=('track','favorites'), data=tracks)

ax = plt.subplot2grid(fig_dims, (1, 0))
sns.countplot(x=('album','favorites'), data=tracks)

ax = plt.subplot2grid(fig_dims, (2, 0))
sns.countplot(x=('artist','favorites'), data=tracks)


# In[ ]:





# In[74]:


import lux
import pandas as pd


# In[76]:


tracks

