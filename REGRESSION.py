#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('C:/Users/personal/Desktop/CAT2020/DATASCIENCE/data-master/world-cup-predictions/wc-20140609-140000.csv')


# In[4]:


df.drop(['country','country_id','group'],inplace=True,axis=1)


# In[9]:


df.columns


# In[16]:


X=df.drop('win',axis=1)
Y=df['win']


# In[17]:


from sklearn import linear_model
regg=linear_model.LinearRegression()


# In[18]:


regg.fit(X,Y)


# In[21]:


regg.coef_


# In[23]:


regg.intercept_


# In[32]:


regg.predict(np.array([[63.43,1.1208,1.1636,0.0631,0.2032,0.038517,0.007996,0.001021]]))


# In[53]:


y_PRED=regg.predict(X)


# In[54]:


y_PRED


# In[37]:


regg.score(X,Y)


# In[62]:


from sklearn.linear_model import Lasso
model = Lasso(alpha=0.005)
model.fit(X,Y)


# In[63]:


model.coef_


# In[64]:


model.predict(X)


# In[65]:


parameters= {'alpha':[0.1,0.01,0.005,0.001,0.0001,0.5]}
from sklearn.model_selection import GridSearchCV


# In[66]:


LL=Lasso()


# In[68]:


Grid1=GridSearchCV(LL,parameters)
Grid1.fit(X,Y)


# In[69]:


Grid1.best_estimator_

