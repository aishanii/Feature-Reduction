#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as numpy
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[31]:


#importing the dataset
df = pd.read_csv(r'C:\Users\Admin\Downloads\redwinequality.csv')
df.head()


# ### Here, we have 11 predictors or "features", on the basis of which we predict the wine quality between 0-10.

# In[34]:


X = df.drop('quality',axis=1)
y = df['quality']


# In[35]:


chi_scores = chi2(X,y)


# In[36]:


chi_scores


# ### Here, chi-square is represented by the first array, and the second array represents the *p-value*. P-value, in very simple words, can be determined to find out how much the output feature is affected by a certain feature, how far away it is from the mean. 

# In[37]:


p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)


# In[38]:


p_values.plot.bar()


# ### Here, we can see that both "density" and "pH" values have high and more or less the same p-values, and do not affect the output feature much, hence can be discarded. 

# In[ ]:




