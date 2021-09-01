#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\Admin\Downloads\redwinequality.csv')
df.head()


# ### Applying standardization so that a few high-ranged features don't dominate.

# In[10]:


from sklearn.preprocessing import StandardScaler
variables = ['fixed acidity', 'volatile acidity', 'citric acid', 
             'residual sugar',
             'chlorides','free sulfur dioxide','total sulfur dioxide',
             'density','pH','sulphates','alcohol','quality'
            ]
x = df.loc[:,variables].values
y = df.loc[:,['quality']].values
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)
x.head()


# In[11]:


#importing PCA module 
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)
x_pca = pd.DataFrame(x_pca)
x_pca.head()


# In[13]:


pca.explained_variance_ratio_ #returns a vector of variance


# In[24]:


x_pca['quality']=y
x_pca.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','quality']
x_pca.head()


# In[ ]:




