#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np


# In[8]:


df=pd.read_csv('C:/Users/personal/Desktop/CAT2020/DATASCIENCE/breast_cancer.csv')


# In[11]:


df['diagnosis']=df['diagnosis'].apply(lambda x: 1 if x=='M' else 2)


# In[12]:


df


# In[23]:



X = np.nan_to_num(df.values[:,1:])


# In[24]:


from sklearn.cluster import KMeans 
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[25]:


df['labels']=labels


# In[35]:


df


# In[37]:


import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('radius')
ax.set_ylabel('texure')
ax.set_zlabel('permiter')

ax.scatter(X[:, 1], X[:, 2], X[:, 3], c= labels.astype(np.float))

