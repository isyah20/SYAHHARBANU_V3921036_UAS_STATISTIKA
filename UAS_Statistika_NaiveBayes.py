#!/usr/bin/env python
# coding: utf-8

# In[23]:


#multidimentional array
import numpy as np
#manipulation and analize data
import pandas as pd
#visualization
import matplotlib
import matplotlib.pyplot as plt
#feature engineering
import sklearn
#menampilkan data dalam grafik
import seaborn as sns


# In[24]:


#menampilkan data
dataset = pd.read_csv('Jan_2019_ontime.csv')
dataset.head()


# In[22]:


#menghapus data yang tidak dipilih menjadi fitur
df = dataset.drop(columns=['DAY_OF_MONTH','DAY_OF_WEEK','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER',
                           'ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID','DEP_TIME','DEP_TIME_BLK',
                           'DEP_DEL15','ARR_TIME','ARR_DEL15','DIVERTED','DISTANCE'])
data = df.head(5000)
data #untuk menampilkan data dalam bentuk tabel


# In[25]:


#fitur yang dipilih
data.keys()


# In[36]:


#untuk memisahkan data x dan data y
X = data.iloc[:, :4].values#mengambil 4 dataset untuk confusion matrix
Y = data.iloc[:, 4].values


# In[46]:


#aturan dari gussian naive bayes data harus numeric
#untuk mengubah data yang kategorikal menjadi numerik 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])


# In[47]:


#dataset sudah diubah menjadi numerik
print(X)


# In[48]:


#membagi dataset menjadi 2 (training dan testing)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

print(x_test)


# In[49]:


#melakukan normalisasi data dengan scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print(x_train)


# In[50]:


#Metode klasifikasi naive bayes yang digunakan yaitu gussian nb
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train, y_train)


# In[51]:


#menampilkan data prediksi
y_pred = classifier.predict(x_test)

print(y_pred)


# In[52]:


#Cek nilai akurasi
from sklearn.metrics import confusion_matrix,accuracy_score

ac = accuracy_score(y_test, y_pred)
print(ac)


# In[55]:


#menampilkan confusion matriks
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[56]:


#menampilkan confusion matriks dalam grafik
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[ ]:




