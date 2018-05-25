
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet = pd.read_csv('Position_Salaries.csv')
lvl = dataSet.iloc[:, 1:2].values
slry = dataSet.iloc[:, 2].values


# In[2]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(lvl, slry)


# In[4]:


print(regressor.predict(lvl))


# In[5]:


# DTR takes average of dependednt values in the splits that it has made.

