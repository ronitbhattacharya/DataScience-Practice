#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[95]:


df=pd.read_csv('C:/Users/personal/Desktop/CAT2020/DATASCIENCE/breast_cancer.csv')


# In[36]:


df.columns


# In[15]:


df.isnull().sum()


# In[40]:


df


# In[16]:


df['compactness_mean'].mean()


# In[17]:


df['compactness_mean'].replace(to_replace=np.nan,value=df['compactness_mean'].mean(),inplace=True)


# In[19]:


df.fillna(0,inplace=True)


# In[25]:


df1=df.drop(labels='diagnosis',axis=1)


# In[27]:


from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
scaled_df=scaling.fit_transform(df1)


# In[33]:


df_Scaled=pd.DataFrame(scaled_df)
df_Scaled.drop(labels=0,axis=1,inplace=True)


# In[37]:



headers=['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'Cancer_y_n']


# In[38]:


df_Scaled.columns=headers


# In[41]:


df_Scaled.drop('Cancer_y_n',axis=1,inplace=True)


# In[42]:


df_Scaled


# In[51]:


df_Scaled=df_Scaled.join(df['diagnosis'])


# In[55]:


df_Scaled=df_Scaled.join(df['Cancer_y_n'])


# In[56]:


df_Scaled


# In[57]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df_Scaled['diagnosis']= label_encoder.fit_transform(df_Scaled['diagnosis'])


# In[58]:


df_Scaled


# In[63]:


df1.drop(labels='id',axis=1,inplace=True)


# In[110]:


df['Cancer_y_n']=df['Cancer_y_n'].astype(dtype=str)


# In[115]:


df.dtypes


# In[131]:


df.groupby(['Cancer_y_n','diagnosis'])['radius_mean'].mean()


# In[165]:


df.groupby(['Cancer_y_n','diagnosis']).size().reset_index(name='count')


# In[176]:


data= {'B': [184,173,357],'M':[115,97,212],'Total':[299,270,569]}
cont_table=pd.DataFrame.from_dict(data)
cont_table.rename(index={0: '0', 1: '1', 2: 'Total'},inplace=True)


# In[182]:


cont_table.drop('Total',axis=1,inplace=True)


# In[185]:


cont_table.drop('Total',axis=0,inplace=True)


# In[186]:


cont_table


# In[187]:


import scipy
scipy.stats.chi2_contingency(cont_table,correction=True)


# In[170]:


import matplotlib.pyplot as plt
x=df['radius_mean']
y=df['radius_worst']
plt.scatter(x,y)


# In[171]:


df.corr()

