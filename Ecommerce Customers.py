#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


file=pd.read_csv("Ecommerce Customers.csv")


# In[3]:


file


# In[4]:


file.info()


# In[5]:


file.isnull().sum()


# In[6]:


file.duplicated()


# In[7]:


file.drop_duplicates()


# In[8]:


file.describe()


# In[9]:


file.corr()


# In[46]:


sns.heatmap(file.corr(),  annot= True)


# In[10]:


sns.pairplot(file)


# In[11]:


file.columns


# In[12]:


x=file[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y=file[['Yearly Amount Spent']]


# In[13]:


x


# In[14]:


y


# In[42]:


sns.distplot(file['Avg. Session Length'])


# In[43]:


sns.distplot(file['Time on App'])


# In[44]:


sns.distplot(file['Time on Website'])


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=1)


# In[16]:


x_train


# In[17]:


x_test


# In[18]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[19]:


y_pred=lr.predict(x_test)


# In[20]:


y_pred


# In[21]:


plt.scatter(y_pred,y_test,color='red')


# In[47]:


plt.savefig("ecommerce.png")


# In[22]:


from sklearn import metrics


# In[23]:


metrics.r2_score(y_pred,y_test)


# In[24]:


lr.score(x,y)


# In[25]:


from sklearn.linear_model import LinearRegression,Ridge


# In[26]:


rl=Ridge(alpha=0.3)


# In[27]:


rl.fit(x_train,y_train)


# In[28]:


rl.predict(x_test)


# In[29]:


metrics.r2_score(rl.predict(x_test),y_test)


# In[30]:


plt.scatter(rl.predict(x_test),y_test)


# In[ ]:




