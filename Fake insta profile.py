#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score,roc_curve,confusion_matrix


# In[4]:


from jupyterthemes import jtplot
jtplot.style(theme='monokai',context='notebook',ticks=True,grid=False)


# In[6]:


instagram_df_train=pd.read_csv('insta_train.csv')
instagram_df_train


# In[7]:


instagram_df_test=pd.read_csv('insta_test.csv')
instagram_df_test


# In[9]:


instagram_df_train.head()


# In[10]:


instagram_df_train.tail()


# In[11]:


instagram_df_test.head()


# In[13]:


instagram_df_test.tail()


# In[14]:


instagram_df_train.info()


# In[15]:


instagram_df_train.describe()


# In[16]:


instagram_df_train.isnull().sum()   #checking if there are null elements


# In[17]:


instagram_df_train['profile pic'].value_counts()


# In[18]:


instagram_df_train['fake'].value_counts()


# In[19]:


#visualizing the data
sns.countplot(instagram_df_train['fake'])


# In[20]:


sns.countplot(instagram_df_train['private'])


# In[21]:


sns.countplot(instagram_df_train['profile pic'])


# In[22]:


plt.figure(figsize=(20,10))
sns.distplot(instagram_df_train['nums/length username'])


# In[23]:


plt.figure(figsize=(20,20))
sns.pairplot(instagram_df_train)


# In[24]:


#finding correlation between features in training data set
plt.figure(figsize=(20,20))
cm=instagram_df_train.corr()
ax=plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)


# In[27]:


sns.countplot(instagram_df_test['fake'])


# In[28]:


sns.countplot(instagram_df_test['profile pic'])


# In[29]:


sns.countplot(instagram_df_test['private'])


# In[30]:


X_train=instagram_df_train.drop(columns=['fake'])
X_test=instagram_df_test.drop(columns=['fake'])
X_train


# In[31]:


X_test


# In[32]:


y_train=instagram_df_train['fake']
y_test=instagram_df_test['fake']
y_train


# In[33]:


#scaling/normalizing the data
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler_x=StandardScaler()
X_train=scaler_x.fit_transform(X_train)
X_test=scaler_x.transform(X_test)


# In[34]:


#converting a single column matrix for fake column to matrix with two columns
#where [1,0] denotes fake and [0,1] denotes real account
y_train=tf.keras.utils.to_categorical(y_train,num_classes=2)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=2)


# In[35]:


y_train


# In[36]:


y_test


# In[85]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[86]:


Training_data=len(X_train)/(len(X_train)+len(X_test))*100
Training_data


# In[87]:


Test_data=len(X_test)/(len(X_train)+len(X_test))*100
Test_data


# In[88]:


import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model=Sequential()
model.add(Dense(50,input_dim=11,activation='relu'))
model.add(Dense(150,activation='relu'))   #input dimensions need to be specified only once. later on its automatically done
model.add(Dense(150,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dropout(0.3))#dropout layer ensures computer to generalise and not memorise
model.add(Dense(2,activation='softmax'))
model.summary()


# In[89]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[90]:


epochs_hist=model.fit(X_train,y_train,epochs=20,verbose=1,validation_split=0.1)


# In[91]:


print(epochs_hist.history.keys())


# In[92]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progression during Training/Validation')
plt.ylabel('Training and validation losses')
plt.xlabel("Epoch number")
plt.legend(['Training Loss','Validation Loss'])


# In[93]:


predicted=model.predict(X_test)


# In[94]:


predicted_value=[]
test=[]
for i in predicted:
    predicted_value.append(np.argmax(i))
for i in y_test:
    test.append(np.argmax(i))


# In[95]:


print(classification_report(test,predicted_value))


# In[96]:


plt.figure(figsize=(10,10))
cm=confusion_matrix(test,predicted_value)
sns.heatmap(cm,annot=True)


# In[ ]:




