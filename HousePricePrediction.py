#!/usr/bin/env python
# coding: utf-8

# # price predictions

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_excel('C:/Users/personal/Desktop/MBA/DataScience Git/Houseprice data.xlsx')


# In[9]:


df['CHAS'].value_counts().to_frame()


# In[49]:


df.columns


# In[10]:


df['RAD'].value_counts().to_frame()


# In[15]:


df.describe()


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


import matplotlib.pyplot as plt
df.hist(bins=50,figsize=(20,15))


# In[27]:


corr_matrix=df.corr()


# In[29]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[34]:


from sklearn.impute import SimpleImputer
import numpy as np


# In[35]:


imputer=SimpleImputer(missing_values=np.nan,
    strategy='median')
imputer.fit(df)


# In[36]:


X=imputer.transform(df)


# In[39]:


df_withoutnull=pd.DataFrame(X,columns=df.columns)
df_withoutnull


# In[44]:


from sklearn.preprocessing import StandardScaler


# In[45]:


df_Scaled=df_withoutnull.drop(['CHAS','RAD','MEDV'],axis=1)


# In[52]:


scalar=StandardScaler()
df_Scaled_notnull=pd.DataFrame(scalar.fit_transform(df_Scaled),columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS',  'TAX',
       'PTRATIO', 'Btown', 'LSTAT'])


# In[58]:


df_Scaled_final=df_Scaled_notnull.join(df[['CHAS','RAD','MEDV']])


# In[60]:


df_Scaled_final


# In[61]:


X=df_Scaled_final.drop('MEDV',axis=1)
Y=df_Scaled_final['MEDV']


# In[62]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=12345)


# In[139]:


#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[140]:


#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()


# In[141]:


model.fit(X_train,Y_train)


# In[142]:


Y_pred=model.predict(X_train)


# In[143]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_train,Y_pred)
rmse=np.sqrt(mse)


# In[144]:


rmse


# cross validation #To avoid overfitting of the data

# In[145]:


from sklearn.model_selection import cross_val_score


# In[146]:


scores=cross_val_score(model,X_train,Y_train,scoring="neg_mean_squared_error",cv=10)
rmse_squares=np.sqrt(-scores)


# In[147]:


rmse_squares


# In[148]:


def print_Scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Std Dev", scores.std())


# In[149]:


print_Scores(rmse_squares)


# In[152]:


from joblib import dump,load
dump(model,'PricesHouse.joblib')


# In[157]:


from joblib import dump,load
model= load('PricesHouse.joblib')
import numpy as np
model.predict(np.array([[-0.415377,2.516647,-1.298123,-1.336307,0.835371,-0.752992,1.917048,-0.298373,-1.690182,0.374827,-0.922773,0,5]]))

