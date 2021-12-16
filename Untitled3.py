#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


pima=pd.read_csv('diabetes.csv')
pima.head()


# In[3]:


x=list(pima.columns)
print('list of attributes :',x)


# In[4]:


x.remove('Outcome')
print('pridicted attributes:',x)


# In[5]:


x=pima[pima.columns[:-1]]
y=pima['Outcome']
print(x)
print(y)


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
print(x_train)


# In[7]:


clf=DecisionTreeClassifier()
clf=clf.fit(x,y)


# In[11]:


from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib
dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True, rounded=True,special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[9]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[10]:


from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# In[ ]:




