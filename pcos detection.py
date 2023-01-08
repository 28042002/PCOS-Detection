#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from warnings import filterwarnings
filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#read datasets
df_inf=pd.read_csv("C:\\Users\\chall\\Downloads\\PCOS_infertility.csv", encoding='iso-8859-1')
df_noinf=pd.read_csv("C:\\Users\\chall\\Downloads\\data without infertility _final.csv", encoding='iso-8859-1')
print(f"Shape of df_inf:{df_inf.shape}")
print(f"Shape of df_noinf:{df_noinf.shape}")


# In[3]:


df_inf.sample(5)


# In[4]:


df_noinf.sample(5)


# In[5]:


corr_features=df_noinf.corrwith(df_noinf["PCOS (Y/N)"]).abs().sort_values(ascending=False)
#features with correlation more than 0.4
corr_features=corr_features[corr_features>0.4].index
corr_features


# In[6]:


df_inf.corrwith(df_inf["PCOS (Y/N)"]).abs()


# In[7]:


df_noinf=df_noinf[corr_features]
df_noinf.head()


# In[8]:


df_noinf.columns


# In[10]:


plt.figure(figsize=(14,5))
plt.subplot(1,6,1)
sns.boxplot(x='PCOS (Y/N)',y='Follicle No. (R)',data=df_noinf)
#plt.subplot(1,7,2)
#sns.boxplot(x='PCOS (Y/N)',y='Insulin levels (æIU/ml)',data=df_noinf)
plt.subplot(1,6,2)
sns.boxplot(x='PCOS (Y/N)',y='Follicle No. (L)',data=df_noinf)
plt.subplot(1,6,3)
sns.boxplot(x='PCOS (Y/N)',y='Skin darkening (Y/N)',data=df_noinf)
plt.subplot(1,6,4)
sns.boxplot(x='PCOS (Y/N)',y='hair growth(Y/N)',data=df_noinf)
plt.subplot(1,6,5)
sns.boxplot(x='PCOS (Y/N)',y='Weight gain(Y/N)',data=df_noinf)
plt.subplot(1,6,6)
sns.boxplot(x='PCOS (Y/N)',y='Cycle(R/I)',data=df_noinf)
plt.show()


# In[11]:


plt.figure(figsize=(6,5))
sns.heatmap(df_noinf.corr(), annot=True)
plt.show()


# In[12]:


y=df_noinf['PCOS (Y/N)']
X=df_noinf.drop(['PCOS (Y/N)'], axis=1)


# In[13]:


X_train,X_test,y_train, y_test=train_test_split(X,y, test_size=0.2)


# In[14]:


model=LogisticRegression()
model.fit(X_train,y_train)
print(f"Score in Train Data : {model.score(X_train,y_train)}")


# In[15]:


y_pred=model.predict(X_test)


# In[16]:


print(f"Score in Test Data : {model.score(X_test,y_test)}")

cm=confusion_matrix(y_test, y_pred)
p_right=cm[0][0]+cm[1][1]
p_wrong=cm[0][1]+cm[1][0]

print(f"Right classification : {p_right}")
print(f"Wrong classification : {p_wrong}")
cm


# In[ ]:




