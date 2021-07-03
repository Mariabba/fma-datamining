#!/usr/bin/env python
# coding: utf-8

# In[150]:


# pandas libraries
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import IPython.display as ipd
import missingno as mso
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
#plt.rcParams['figure.figsize'] = (17, 5)


# In[132]:


eco = pd.read_csv(r'C:\Users\jigok\OneDrive\Desktop\UniPISA\2semestre1Anno\DataMining2\Progetto\Data\fma_metadata\echonest.csv',index_col=0, header=[0, 1, 2])


# In[133]:


eco.info()


# In[134]:


eco.describe()


# In[135]:


eco.head(10)


# In[136]:


print('{1} features for {0} tracks'.format(*eco.shape))

ipd.display(eco['echonest', 'temporal_features'].head(10))


# In[137]:


mso.matrix(eco['echonest','temporal_features'])


# In[138]:


eco.plot()


# In[139]:



x = eco.loc[139, ('echonest', 'temporal_features')]
y = eco.loc[2, ('echonest', 'temporal_features')]

x.plot()
y.plot()
plt.show()


# In[155]:


y.plot()
plt.plot(y.rolling(window=12).mean())
plt.show()


# In[141]:


eco['echonest','temporal_features'].columns


# In[142]:


eco['echonest', 'temporal_features'].plot()


# In[145]:


np.squeeze(eco['echonest', 'temporal_features']).shape


# # Eco- K-Means

# In[151]:


plt.plot(np.squeeze(eco['echonest', 'temporal_features']).T)
plt.show()


# In[156]:



km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0)
km.fit(eco['echonest', 'temporal_features'])


# In[157]:


km.cluster_centers_.shape


# In[158]:


plt.plot(np.squeeze(km.cluster_centers_).T)
plt.show()


# In[ ]:


km_dtw = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, random_state=0)
km_dtw.fit(eco['echonest', 'temporal_features'])
km_dtw.cluster_centers_.shape
plt.plot(np.squeeze(km_dtw.cluster_centers_).T)
plt.show()

