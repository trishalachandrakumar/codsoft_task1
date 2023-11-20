#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier


# In[2]:


df = pd.read_csv('D://Trishala//CodSoft//Titanic//tested.csv')
df.head()


# In[3]:


df.Age.fillna(value=df.Age.median(),inplace=True)
df.dropna(inplace=True)
df.Age = df.Age.astype(int)
df.Fare = round(df.Fare,2)
df = pd.concat([df,pd.get_dummies(df['Embarked'])], axis=1)
df.drop(['Cabin','PassengerId', 'Name', 'Ticket', 'Embarked', 'C'], axis=1, inplace=True)
enc = LabelEncoder()
df.Sex = enc.fit_transform(df.Sex)
df.head()


# In[4]:


df.Sex.value_counts().plot.bar(df.Sex)


# In[5]:


df.Survived.value_counts().plot.bar(df.Survived)


# In[6]:


sns.countplot(x='Survived', data=df, hue='Sex')


# In[7]:


df.Pclass.value_counts().plot.bar(df.Pclass)


# In[8]:


sns.countplot(x='Survived', data=df, hue='Pclass')


# In[9]:


df.SibSp.value_counts().plot.bar(df.SibSp)


# In[10]:


sns.countplot(x='Survived', data=df, hue='SibSp')


# In[11]:


df.Parch.value_counts().plot.bar(df.Parch)


# In[12]:


sns.countplot(x='Survived', data=df, hue='Parch')


# In[13]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='Greens')


# In[14]:


sns.pairplot(df)


# In[15]:


x = df.iloc[:,1:]
y = df.iloc[:,0]
feat = ExtraTreesClassifier()
feat.fit(x,y)
feat.feature_importances_
feat_imp = pd.Series(feat.feature_importances_, index=x.columns)
feat_imp.nlargest(8).plot(kind='barh')


# In[16]:


skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(x,y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[17]:


classifier = LogisticRegression()
classifier.fit(x_train,y_train)


# In[18]:


y_pred = classifier.predict(x_test)
final = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
final.head()


# In[19]:


confusion_matrix(y_test, y_pred)


# In[20]:


plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid()
plt.plot([min(y_test),max(y_test)],[min(y_pred),max(y_pred)], color='red')
plt.title('Actual V/S Predicted')


# In[21]:


accuracy_score(y_test, y_pred)

