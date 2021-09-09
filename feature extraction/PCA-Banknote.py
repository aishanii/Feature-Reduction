#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\Admin\Downloads\data_banknote_authentication.csv')
df.head()


# ### Applying standardization so that a few high-ranged features don't dominate.

# In[3]:


from sklearn.preprocessing import StandardScaler
variables = ['Variance','Skewness','Kurtosis','Entropy']
x = df.loc[:,variables].values
y = df.loc[:,['Class']].values
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)
x.head()


# In[4]:


#importing PCA module 
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)
x_pca = pd.DataFrame(x_pca)
x_pca.head()


# ### Notice that there are four features in the original data and so four principal components are generated.

# In[12]:


pca.explained_variance_ratio_ #retuns an array of variances


# ### Clearly, the first principal component has the highest variance value.  

# In[8]:


#labelling the principal components
x_pca['Class']=y
x_pca.columns = ['PC1','PC2','PC3','PC4','Class']
x_pca['Class'] = LabelEncoder().fit_transform(x_pca['Class'])
x_pca.head()


# In[11]:



fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1') 
ax.set_ylabel('Principal Component 2') 
ax.set_title('2 component PCA') 
targets = [1,0]
colors = ['r', 'g']
for Class, color in zip(targets,colors):
    indicesToKeep = x_pca['Class'] == Class
    ax.scatter(x_pca.loc[indicesToKeep, 'PC1']
    , x_pca.loc[indicesToKeep, 'PC2']
    , c = color
    , s = 50)
ax.legend(targets)
ax.grid()


# In[ ]:




