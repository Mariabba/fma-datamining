#!/usr/bin/env python
# coding: utf-8

# In[2]:


# general libraries
import os
import sys
import math
import statistics
import collections
import missingno as msno
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


# In[3]:


sns.set()


# In[4]:


tracks = pd.read_csv(r'C:\Users\jigok\OneDrive\Desktop\UniPISA\2semestre2Anno\DataMining2\Progetto\Data\fma_metadata\tracks.csv',low_memory=False,index_col=0, header=[0, 1])

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


# In[5]:


tracks.info()


# In[96]:


ipd.display(tracks['track'].head(10))
ipd.display(tracks['album'].head(10))
ipd.display(tracks['artist'].head(10))
ipd.display(tracks['set'].head(10))


# In[7]:


top_data = msno.nullity_filter(tracks, filter='bottom', n= 50)


# # Looking the single column value

# In[8]:


tracks[("album", "tags")]


# In[ ]:





# In[9]:


tracks[("album", "favorites")]


# In[10]:


plt.hist(tracks[("album", "favorites")])


# In[81]:


tracks[("album", "favorites")].value_counts()


# In[25]:


tracks[("track", "favorites")]


# In[47]:


fig =plt.subplots(figsize=(50, 20))
ax = plt.subplot2grid(fig_dims, (0, 0))
sns.distplot(tracks[("track", "favorites")].dropna(), kde=False)
plt.ylabel("count")


# In[83]:


tracks[("track", "favorites")].value_counts()


# In[49]:


fig =plt.subplots(figsize=(50, 20))
ax = plt.subplot2grid(fig_dims, (0, 0))
sns.countplot(x=('track','favorites'), data=tracks)


# In[11]:


tracks[("track", "genre_top")]


# In[73]:


fig =plt.subplots(figsize=(40, 20))
ax = sns.countplot(y=("track", "genre_top"), data=tracks, palette='hls')
plt.yticks(fontsize=30)
plt.xticks(fontsize=40)
plt.show()


# In[85]:


tracks[("track", "genre_top")].value_counts()


# In[13]:


tracks[("album","id")]


# In[14]:


plt.hist(tracks[("album", "id")])


# In[15]:


tracks[("artist", "name")]


# In[16]:


tracks[("artist", "name")].unique()


# In[86]:


tracks[("artist", "name")].value_counts()


# In[17]:


tracks[("album", "type")]


# In[18]:


sns.countplot(y=("album", "type"), data=tracks, palette='hls')
plt.title('Frequency of album type')
plt.show()


# In[80]:


tracks[("album", "type")].value_counts()


# In[87]:


tracks[("album", "listens")].value_counts()


# In[92]:


fig =plt.subplots(figsize=(20, 20))
ax = plt.subplot2grid(fig_dims, (0, 0))
sns.distplot(tracks[("album", "listens")].dropna(), kde=False)
plt.ylabel("count")


# In[89]:


tracks[("track", "listens")].value_counts()


# In[94]:


fig =plt.subplots(figsize=(20, 10))
ax = plt.subplot2grid(fig_dims, (0, 0))
sns.distplot(tracks[("track", "listens")].dropna(), kde=False)
plt.ylabel("count")


# # Missing Value detection

# In[19]:


msno.matrix(top_data)


# In[20]:


fig = plt.subplots(figsize=(15, 5))
fig_dims = (1, 1)

ax = plt.subplot2grid(fig_dims, (0, 0))
msno.matrix(tracks, ax=ax,sparkline=False)
plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=5)
plt.plot()


# In[21]:


msno.bar(top_data)


# In[22]:


msno.heatmap(tracks)


# In[24]:


# checking columns' missing data
column_names = list(tracks.columns)
for column_name in column_names:
    # count number of rows with missing values
    data = tracks[column_name]
    n_miss = data.isnull().sum()
    perc = n_miss / tracks.shape[0] * 100
    if n_miss > 0:
        print( '%s, Missing: %d (%.1f%% of rows) ' % (column_name, n_miss, perc))


# In[ ]:





# # Zero Value Detection

# In[53]:


data_types = tracks.dtypes
for column_name, column_type in data_types.items():
    if column_type == np.int or column_type == np.float:
        zero_values = (tracks[column_name] == 0).sum()
        if zero_values > 0:
            print(column_name, zero_values, sep="\t")


# Gli attributi numerici possono presentare missing value ma non ci sono in questo caso, ottimo

# # Controllo se ci sono Rows duplicate

# # Columns Unique Value Lookup

# Elimino le colonne con valori strani tipo (album, tags) prima, senno mi da errore

# In[54]:


tracks.info()


# In[55]:


#del tracks["album", "tags"]
#del tracks["artist", "tags"]
#del tracks["track", "tags"]


# In[56]:


column_names = tuple(tracks.columns)
for column_name in column_names: 
    unique_values = tracks[column_name].unique()
    if len(unique_values) >= 10:
        print(column_name, "more than 10 %s values" % tracks.dtypes[column_name], sep="\t")
    else:
         print(column_name, unique_values, sep="\t")


# # Correlation

# In[57]:


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


# In[58]:


# compute the correlation matrix, excluding NA/null values
correlation_matrix = tracks.corr("pearson")  # pearson correlation
# draw heatmap
draw_sns_heatmap(correlation_matrix, 0, 250, "Pearson correlation matrix", "pearson_correlation_matrix.png")


# # First Look Distribution

# In[75]:


fig = plt.subplots(figsize=(20, 20))
fig_dims =(3,1)

ax = plt.subplot2grid(fig_dims, (0, 0))
sns.countplot(x=('track','favorites'), data=tracks)

ax = plt.subplot2grid(fig_dims, (1, 0))
sns.countplot(x=('album','favorites'), data=tracks)

ax = plt.subplot2grid(fig_dims, (2, 0))
sns.countplot(x=('artist','favorites'), data=tracks)


# In[ ]:




