#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


from sklearn import datasets


# In[3]:


from sklearn import svm
from sklearn import metrics


# In[4]:


cancer=datasets.load_breast_cancer()


# In[5]:


print(cancer.feature_names)


# In[6]:


print(cancer.target_names)


# In[7]:


x=cancer.data


# In[8]:


y=cancer.target


# In[9]:


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)


# In[10]:


print(x_train,y_train)


# In[11]:


classes= ['malignent','beningn']


# In[12]:


clf = svm.SVC(kernel="linear")


# In[13]:


clf.fit(x_train,y_train)


# In[19]:


predict = clf.predict(x_test)


# In[22]:


acc = metrics.accuracy_score(y_test,predict)


# In[23]:


print(acc)


# In[35]:


for i in range(len(x_test)):
    print("predicted data: ",classes[predict[i]],"    actual data:   ",classes[y_test[i]])
    
    
    


# In[ ]:





# In[ ]:




