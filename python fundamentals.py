#!/usr/bin/env python
# coding: utf-8

# Python basics

# In[6]:


a =4.5 #type casting 
print(int(a))


# In[8]:


A="Ronit"
print(A[-2])


# In[10]:


A="Ronit"
B="Mohika"
print(A[0:4]+B[0:2])


# In[13]:


A="Ronit"
print(A[0:5:2])


# In[18]:


A=A.replace('Ronit','Mohika')
print(A)


# In[21]:


AA=(1,2,"Ronit","Mohika",4.0)
print(AA[0:3])


# In[26]:


AA=list(AA)    #lists are mutable tuples are immutable 
AA[2]="Bratendu"
AA


# In[29]:


AA.append(['Ronit'])#append will add only one element 
AA.extend(['Pradyumna','Suman'])
print(AA)


# In[33]:


#split operator 
A="Ronit B"
L=[]
L=A.split(' ')
print(L)


# In[53]:


A=[{"a":1},{"b":2},{"c":3}]
import pandas as pd
df=pd.DataFrame(A)
df


# In[57]:


dict_aa={u'2012-06-08': 388,
 u'2012-06-09': 388,
 u'2012-06-10': 388,
 u'2012-06-11': 389,
 u'2012-06-12': 389,
 u'2012-06-13': 389,
 u'2012-06-14': 389,
 u'2012-06-15': 389,
 u'2012-06-16': 389,
 u'2012-06-17': 389,
 u'2012-06-18': 390,
 u'2012-06-19': 390,
 u'2012-06-20': 390,
 u'2012-06-21': 390,
 u'2012-06-22': 390}


# In[60]:


df=pd.DataFrame(dict_aa.items(),columns=['Date','Value'])
df


# In[63]:


LL=['a','b','b',2.0]
LL_NEW=set(LL)
LL_NEW  #sets will remove duplicate values 


# In[66]:


work_Ex=int(input('Enter your work experience'))
if(work_Ex>2):
    print("Live project + internship")
elif(work_Ex==2):
    print("Nothing")
else:
    print("Internship")


# In[94]:


aA=list(range(5,10,2))
print(aA)


# In[124]:


string ='Ronit is lit'.split(' ')
new_list=[]
for i in range(0,len(string)):
    for j in range(0,len(string[i])):
        new_list.append(string[i][j])
        
print(new_list)
        


   


# In[132]:


list_aa=list(range(0,10))
list_aa=map(lambda x:x*2, list_aa)
print(list(list_aa))


# In[140]:


listaa=[1,2,3,4,5,6]

        
def multp(a):
    for i in range(0,len(a)):
        a[i]=a[i]*2
    return(a)        
     
print(multp(listaa))


# In[143]:


try:
    a=int(input())
    print(a*2)
except:
    print('enter no only')


# In[144]:


class Points(object):
  def __init__(self,x,y):

    self.x=x
    self.y=y

  def print_point(self):

    print('x=',self.x,' y=',self.y)

p1=Points(1,2)
p1.print_point()


# In[145]:


class Points(object):

  def __init__(self,x,y):

    self.x=x
    self.y=y

  def print_point(self):

    print('x=',self.x,' y=',self.y)

p2=Points(1,2)

p2.x='A'

p2.print_point()


# In[150]:


AAA={'A':[1,2,3],'B':[4,5,6]}
import pandas as pd
df_new=pd.DataFrame(AAA)
df_new


# In[172]:


df=pd.read_csv('C:/Users/personal/Desktop/CAT2020/DATASCIENCE/breast_cancer.csv')
df.describe(include='all')


# In[182]:


df


# In[181]:


df['radius_mean'][df['diagnosis']=='B'].mean()


# In[190]:


df.iloc[0:3,0:3]


# In[198]:


list_aa=[1,2,3,4]
import numpy as np
arr=np.array(list_aa)
arr.max()


# In[208]:


arr_new=np.linspace(0,2*(np.pi),50)
import matplotlib.pyplot as plt
y=np.sin(arr_new)
plt.plot(arr_new,y)


# In[212]:


a=np.array([0,1])
b=np.array([1,0])
np.dot(a,b) 


# In[217]:


import requests
response = requests.get("https://jsonplaceholder.typicode.com/todos")
print(response.status_code)


# In[221]:


data_raw=(response.json())
data_raw


# In[236]:


df_new=pd.DataFrame(data_raw,columns=['userId','id','title','completed'])


# In[248]:


get_ipython().system('pip install pycoingecko')
from pycoingecko import CoinGeckoAPI
bg=CoinGeckoAPI()
bitcoin_data=bg.get_coin_market_chart_by_id(id='bitcoin',vs_currency='usd',days=30)
bitcoin_data


# In[272]:


bitcoin_data_df


# In[279]:


bitcoin_data_df=pd.DataFrame(bitcoin_data)
l=[]
m=[]
for i in range(0,722):
    l=bitcoin_data_df.iloc[i,0]
    m.append(l[0])
    
df_time = pd.DataFrame(m,columns=['TimeStamp'])
df_time['Time']=pd.to_datetime(df_time['TimeStamp'],unit='ms')
df_time


# In[281]:


a=np.array([0,1,0,1,0]) 
b=np.array([1,0,1,0,1]) 
print(a*b) 

