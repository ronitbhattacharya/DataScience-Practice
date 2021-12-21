#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.datasets import fetch_openml


# In[5]:


mnist=fetch_openml('mnist_784')


# In[6]:


x,y=mnist['data'],mnist['target']


# In[42]:


a=x.iloc[3601,:]


# In[26]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=12345)


# In[30]:


Y_train=Y_train.astype('int8')
Y_test=Y_test.astype('int8')


# In[32]:


Y_train=(Y_train==2)
Y_test=(Y_test==2)


# In[33]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)


# In[47]:


y_pred=model.predict(X_train)


# In[48]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_train,y_pred)


# In[50]:


from sklearn.model_selection import cross_val_score
clf=cross_val_score(model,X_train,Y_train,cv=10,scoring="accuracy")


# In[52]:


clf.mean()


# In[55]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train,y_pred)


# In[57]:


from sklearn.metrics import f1_score,precision_score
f1_score(Y_train,y_pred)

