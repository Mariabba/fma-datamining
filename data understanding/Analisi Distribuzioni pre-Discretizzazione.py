#!/usr/bin/env python
# coding: utf-8

# In[45]:


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
sns.set()
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


# In[46]:


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


# In[47]:


tracks.info()


# In[48]:


ipd.display(tracks['track'].head())
ipd.display(tracks['album'].head())
ipd.display(tracks['artist'].head())
ipd.display(tracks['set'].head())


# In[49]:


column_names = tuple(tracks.columns)
print("nome tupla:", column_names)


# In[50]:


#prova1
column_names = tuple(tracks.columns)
for column_name in column_names: 
    unique_values = []
    unique_values.append(tracks[column_name])
    print("Colonna:",column_name, unique_values, sep="\t")


# # Visualizzazione Distribuzioni

# # Analisi Valori Interi

# In[51]:


df_mod = tracks.select_dtypes(include=['int64'])
df_mod.dtypes


# Analisi delle distribuzioni con plot-histogram per ALBUM

# In[52]:


fig =plt.subplots(figsize=(30, 20))
fig_dims = (3, 3)

ax = plt.subplot2grid(fig_dims, (0, 0))
sns.distplot(tracks["album", "comments"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (0, 1))
sns.distplot(tracks["album", "favorites"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (1, 0))
sns.distplot(tracks["album", "id"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (1, 1))
sns.distplot(tracks["album", "listens"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (2, 0))
sns.distplot(tracks["album", "tracks"].dropna(), kde=False,bins=20)

plt.show()


# In[53]:


tracks["album", "comments"].value_counts()


# In[54]:


tracks["album", "favorites"].value_counts()


# Analisi delle distribuzioni con plot-histogram per ARTIST

# In[55]:


fig =plt.subplots(figsize=(40, 20))
fig_dims = (3, 3)

ax = plt.subplot2grid(fig_dims, (0, 0))
sns.distplot(tracks["artist", "comments"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (1, 0))
sns.distplot(tracks["artist", "favorites"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (2, 0))
sns.distplot(tracks["artist", "id"].dropna(), kde=False,bins=20)


# Analisi delle distribuzioni con plot-histogram per TRACK

# In[56]:


fig =plt.subplots(figsize=(20, 30))
fig_dims = (4, 2)

ax = plt.subplot2grid(fig_dims, (0, 0))
sns.distplot(tracks["track", "bit_rate"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (0, 1))
sns.distplot(tracks["track", "comments"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (1, 0))
sns.distplot(tracks["track", "duration"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (1, 1))
sns.distplot(tracks["track", "favorites"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (2, 0))
sns.distplot(tracks["track", "interest"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (2, 1))
sns.distplot(tracks["track", "listens"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (3, 0))
sns.distplot(tracks["track", "number"].dropna(), kde=False,bins=20)

plt.show()


# In[57]:


tracks["track", "comments"].value_counts()


# In[58]:


tracks["track", "interest"]


# In[59]:


tracks["track", "interest"].unique()


# In[60]:


tracks["track", "number"].value_counts()


# In[61]:


tracks["track", "number"].unique()


# In[62]:


tracks["track", "listens"].value_counts()


# # Analisi valori FLOAT

# In[63]:


df_mod = tracks.select_dtypes(include=['float64'])
df_mod.dtypes


# In[64]:


fig =plt.subplots(figsize=(30, 20))
fig_dims = (3, 3)

ax = plt.subplot2grid(fig_dims, (0, 0))
sns.distplot(tracks["artist", "latitude"].dropna(), kde=False,bins=20)
ax = plt.subplot2grid(fig_dims, (1, 0))
sns.distplot(tracks["artist", "longitude"].dropna(), kde=False,bins=20)


# # Analisi valori OBJECT

# In[65]:


df_mod = tracks.select_dtypes(include=['object'])
df_mod.dtypes


# Trasformo la variabile object album,engenieer in categorica

# In[66]:


tracks["album", "engineer"].unique()


# In[67]:


tracks["album", "engineer"] = tracks["album", "engineer"].astype("category")


# In[68]:


tracks["album", "engineer"].value_counts()


# In[69]:


fig =plt.subplots(figsize=(40, 200))
sns.countplot(y=("album", "engineer"), data=tracks, palette='hls')
plt.title('Frequency of album engenieer')
plt.yticks(fontsize=20)
plt.xticks(fontsize=40)
plt.show()


# Trasformo la variabile album, producer

# In[70]:


tracks["album", "producer"].unique()


# In[71]:


tracks["album", "producer"] = tracks["album", "producer"].astype("category")


# In[72]:


tracks["album", "producer"].value_counts()


# In[73]:


fig =plt.subplots(figsize=(40, 200))
sns.countplot(y=("album", "producer"), data=tracks, palette='hls')
plt.title('Frequency of album producer')
plt.yticks(fontsize=15)
plt.xticks(fontsize=40)
plt.show()


# Analizzo album,tags

# In[74]:


print(tracks["album", "tags"])


# In[75]:


"".join([str(_) for _ in  tracks["album", "tags"]])


# In[76]:


tracks["album", "tags"].value_counts()


# Analisi Album Title, lo converto in category

# In[77]:


tracks["album", "title"].unique()


# In[78]:


tracks["album", "title"] = tracks["album", "title"].astype("category")


# In[79]:


tracks["album", "title"].value_counts()


# Analist Artist, associated table

# In[80]:


tracks["artist", "associated_labels"].unique()


# In[81]:


tracks["artist", "associated_labels"] = tracks["artist", "associated_labels"].astype("category")


# In[82]:


tracks["artist", "associated_labels"].value_counts()


# In[83]:


fig =plt.subplots(figsize=(40, 300))
sns.countplot(y=("artist", "associated_labels"), data=tracks, palette='hls')
plt.title('Frequency of associated labels')
plt.yticks(fontsize=10)
plt.xticks(fontsize=40)
plt.show()


# Analisi Artist, location

# In[84]:


tracks["artist", "location"].unique()


# In[85]:


tracks["artist", "location"] = tracks["artist", "location"].astype("category")


# In[86]:


tracks["artist", "location"].value_counts()


# In[163]:


#scommentare per avere l'otput

#fig =plt.subplots(figsize=(40, 300))
#sns.countplot(y=("artist", "location"), data=tracks, palette='hls')
#plt.title('Frequency of associated labels')
#plt.yticks(fontsize=10)
#plt.xticks(fontsize=40)
#plt.show()


# Analisi artist, member

# In[88]:


tracks["artist", "members"].unique()


# In[89]:


tracks["artist", "members"] = tracks["artist", "members"].astype("category")
tracks["artist", "members"].value_counts()


# Analisi Artist,name

# In[90]:


tracks["artist", "name"].unique()


# In[91]:


tracks["artist", "name"] = tracks["artist", "name"].astype("category")
tracks["artist", "name"].value_counts()


# Analist artist related project

# In[92]:


tracks["artist", "related_projects"].unique()


# In[93]:


tracks["artist", "related_projects"] = tracks["artist", "related_projects"].astype("category")
tracks["artist", "related_projects"].value_counts()


# Analisi artist, tags (piango)

# In[94]:


print(tracks["artist", "tags"])


# In[95]:


"".join([str(_) for _ in  tracks["artist", "tags"]])


# In[96]:


tracks["artist", "tags"].value_counts()


# Analisi artist, website

# In[97]:


tracks["artist", "website"].unique()


# In[98]:


tracks["artist", "website"] = tracks["artist", "website"].astype("category")
tracks["artist", "website"].value_counts()


# Analisi artist, wikipedia

# In[99]:


tracks["artist", "wikipedia_page"].unique()


# In[100]:


tracks["artist", "wikipedia_page"] = tracks["artist", "wikipedia_page"].astype("category")
tracks["artist", "wikipedia_page"].value_counts()


# In[164]:


#scommentare per avere l'output

#fig =plt.subplots(figsize=(40, 300))
#sns.countplot(y=("artist", "wikipedia_page"), data=tracks, palette='hls')
#plt.title('Frequency of associated labels')
#plt.yticks(fontsize=20)
#plt.xticks(fontsize=40)
#plt.show()


# Analisi Set,split, questa si riferisce a quale dataset fa riferimento la traccia quindi sia questa che set,subset proporrei di eliminarle

# In[102]:


tracks["set", "split"].unique()


# In[103]:


tracks["set", "split"] = tracks["set", "split"].astype("category")
tracks["set", "split"].value_counts()


# In[104]:


fig =plt.subplots(figsize=(10, 5))
sns.countplot(x=("set","split"), data=tracks, palette='hls')
plt.title('Frequency of split')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[105]:


tracks["set", "subset"].unique()


# In[106]:


tracks["set", "subset"] = tracks["set", "subset"].astype("category")
tracks["set", "subset"].value_counts()


# In[107]:


fig =plt.subplots(figsize=(10, 10))
sns.countplot(x=("set","subset"), data=tracks, palette='hls')
plt.title('Frequency of subset')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.show()


# Analisi track, composer

# In[108]:


tracks["track", "composer"].unique()


# In[109]:


tracks["track", "composer"] = tracks["track", "composer"].astype("category")
tracks["track", "composer"].value_counts()


# In[166]:


#scommentare per avere l'output

#fig =plt.subplots(figsize=(40, 100))
#sns.countplot(y=("artist", "wikipedia_page"), data=tracks, palette='hls')
#plt.title('Frequency of associated labels')
#plt.yticks(fontsize=10)
#plt.xticks(fontsize=40)
#plt.show()


# Analisi track, geners

# In[111]:


print(tracks["track", "genres"])


# In[112]:


"".join([str(_) for _ in  tracks["track", "genres"]])


# In[113]:


tracks["track", "genres"].value_counts()


# Genres all suppongo sia praticamente molto simile ma più numerosa quindi non la analizzo nel dettaglio

# Analisi Track information

# In[114]:


tracks["track", "information"].unique()


# In[115]:


tracks["track", "information"] = tracks["track", "information"].astype("category")
tracks["track", "information"].value_counts()


# Analisi track,language code

# In[116]:


tracks["track", "language_code"].unique()


# In[117]:


tracks["track", "language_code"] = tracks["track", "language_code"].astype("category")
tracks["track", "language_code"].value_counts()


# In[118]:


fig =plt.subplots(figsize=(30, 10))
sns.countplot(x=("track","language_code"), data=tracks, palette='hls')
plt.title('Frequency of a track in a specific language')
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.show()


# Analisi tracks, lyricist

# In[119]:


tracks["track", "lyricist"].unique()


# In[120]:


tracks["track", "lyricist"] = tracks["track", "lyricist"].astype("category")
tracks["track", "lyricist"].value_counts()


# In[121]:


fig =plt.subplots(figsize=(10, 20))
sns.countplot(y=("track","lyricist"), data=tracks, palette='hls')
plt.title('Frequency of a track lyricist')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# Analisi track, publisher

# In[122]:


tracks["track", "publisher"].unique()


# In[123]:


tracks["track", "publisher"] = tracks["track", "publisher"].astype("category")
tracks["track", "publisher"].value_counts()


# In[124]:


fig =plt.subplots(figsize=(10, 20))
sns.countplot(y=("track","publisher"), data=tracks, palette='hls')
plt.title('Frequency of a track publisher')
plt.yticks(fontsize=10)
plt.xticks(fontsize=15)
plt.show()


# Analisi track, tags

# In[125]:


print(tracks["track", "tags"])


# In[126]:


"".join([str(_) for _ in  tracks["track", "tags"]])


# In[127]:


tracks["track", "tags"].value_counts()


# Analisi Track title

# In[128]:


tracks["track", "title"].unique()


# In[129]:


tracks["track", "title"] = tracks["track", "title"].astype("category")
tracks["track", "title"].value_counts()


# In[ ]:





# # Analisi Valori DATETIME

# In[130]:


df_mod = tracks.select_dtypes(include=['datetime64'])
df_mod.dtypes


# Analisi Album,date_created

# In[131]:


tracks["album", "date_created"].unique()


# In[132]:


tracks["album", "date_created"].value_counts()


# In[133]:


tracks = tracks.sort_values(("album", "date_created"), ascending=True)
plt.hist(tracks["album", "date_created"])
plt.xticks(rotation='horizontal')


# In[134]:


tracks['Month-ADateCreated']=pd.to_datetime(tracks["album", "date_created"]).dt.month
tracks['Year-ADateCreated']=pd.to_datetime(tracks["album", "date_created"], format='%d/%m/%Y %H:%M:%S').dt.year
tracks['Day-ADateCreated']=pd.to_datetime(tracks["album", "date_created"], format='%d/%m/%Y %H:%M:%S').dt.day


# In[135]:


del tracks["album", "date_created"]


# In[136]:


tracks = tracks.sort_values(('Month-ADateCreated'), ascending=True)
plt.hist(tracks['Month-ADateCreated'])
plt.xticks(rotation='horizontal')


# In[137]:


tracks = tracks.sort_values(('Year-ADateCreated'), ascending=True)
plt.hist(tracks['Year-ADateCreated'])
plt.xticks(rotation='horizontal')


# In[138]:


tracks = tracks.sort_values(('Day-ADateCreated'), ascending=True)
plt.hist(tracks['Day-ADateCreated'])
plt.xticks(rotation='horizontal')


# Analisi Album Date,released

# In[139]:


tracks["album", "date_released"].value_counts()


# In[140]:


tracks = tracks.sort_values(("album", "date_released"), ascending=True)
plt.hist(tracks["album", "date_released"])
plt.xticks(rotation='horizontal')


# *Analisi Artist,activeYear Begin

# In[141]:


tracks["artist", "active_year_begin"].value_counts()


# In[142]:


tracks = tracks.sort_values(("artist", "active_year_begin"), ascending=True)
plt.hist(tracks["artist", "active_year_begin"])
plt.xticks(rotation='horizontal')


# Analisi Artist, active year end

# In[143]:


tracks["artist", "active_year_end"].value_counts()


# In[144]:


tracks = tracks.sort_values(("artist", "active_year_end"), ascending=True)
plt.hist(tracks["artist", "active_year_end"])
plt.xticks(rotation='horizontal')


# Analisi Artist, date_created

# In[145]:


tracks["artist", "date_created"].value_counts()


# In[146]:


tracks = tracks.sort_values(("artist", "date_created"), ascending=True)
plt.hist(tracks["artist", "date_created"])
plt.xticks(rotation='horizontal')


# Analisi track, datecreated

# In[147]:


tracks["track", "date_created"].value_counts()


# In[148]:


tracks = tracks.sort_values(("track", "date_created"), ascending=True)
plt.hist(tracks["track", "date_created"])
plt.xticks(rotation='horizontal')


# Analisi track,date registered

# In[149]:


tracks["track", "date_recorded"].value_counts()


# In[150]:


tracks = tracks.sort_values(("track", "date_recorded"), ascending=True)
plt.hist(tracks["track", "date_recorded"])
plt.xticks(rotation='horizontal')


# # Analisi valori CATEGORY

# In[151]:


df_mod = tracks.select_dtypes(include=['category'])
df_mod.dtypes


# Analisi Album, Information

# In[152]:


tracks["album", "information"].value_counts()


# In[153]:


tracks["album", "information"].unique()


# Analisi Album,type

# In[154]:


tracks[("album", "type")]


# In[155]:


tracks[("album", "type")].value_counts()


# In[156]:


sns.countplot(y=("album", "type"), data=tracks, palette='hls')
plt.title('Frequency of album type')
plt.show()


# Analisi artist bio

# In[157]:


tracks["artist", "bio"].value_counts()


# Analisi track,genre_top

# In[158]:


tracks[("track", "genre_top")]


# In[159]:


tracks[("track", "genre_top")].value_counts()


# In[160]:


fig =plt.subplots(figsize=(40, 20))
ax = sns.countplot(y=("track", "genre_top"), data=tracks, palette='hls')
plt.yticks(fontsize=30)
plt.xticks(fontsize=40)
plt.show()


# Analisi track,license

# In[161]:


tracks["track", "license"].value_counts()


# In[167]:


#scommentare per avere l'output

#fig =plt.subplots(figsize=(60, 200))
#ax = sns.countplot(y=("track", "license"), data=tracks, palette='hls')
#plt.yticks(fontsize=30)
#plt.xticks(fontsize=40)
#plt.show()


# In[ ]:




