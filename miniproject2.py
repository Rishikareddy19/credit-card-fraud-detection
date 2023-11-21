#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the necessary packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# In[2]:


#loading the data 
data = pd.read_csv('credit card fraud.csv')


# In[3]:


#describing the data
data.head()


# In[4]:


#describing the data
data.describe()


# In[5]:


#checking whether any null values are present
data.info()


# In[6]:


#determining the fraudulent and not fraud transactions
Fraud = data[data['Class'] == 1]
notfraud = data[data['Class'] == 0]
outlier_fraction = len(Fraud)/float(len(notfraud))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('notfraud Transactions: {}'.format(len(data[data['Class'] == 0])))


# In[7]:


# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[8]:


#dividing the X and the Y from the dataset
X=data.drop(['Class'], axis=1)
Y=data['Class']
print(X.shape)
print(Y.shape)
#getting just the values for the sake of processing (its a numpy array with no columns)
X_data=X.values
Y_data=Y.values


# In[9]:


#training and testing the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, train_size=0.70, test_size=0.30, random_state=1)


# In[10]:


# Instantiate the model to an empty object
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Train the model using 'fit' method
model.fit(X_train, y_train)

# Test the model using 'predict' method
y_pred = model.predict(X_test)

# Print the classification report 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[11]:


#printing the confusion matrix
LABELS = ['Notfraud', 'fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[15]:


# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# predictions
y_pred = rfc.predict(X_test)
# Print the classification report 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[16]:


#printing the confusion matrix
LABELS = ['Notfraud', 'fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




