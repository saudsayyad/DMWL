
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# In[2]:


# Importing the dataset
dataset = pd.read_csv('DriverData.csv')
X = dataset.iloc[:, [2, 3]].values


# In[27]:


dataset


# In[28]:


dataset.info()


# In[3]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


print (X)


# In[7]:


dataset.isnull().sum()


# In[8]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[10]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# In[11]:


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Drivers')
plt.xlabel('Concentration Score (Score1)')
plt.ylabel('Response Score (Score2)')
plt.legend()
plt.show()

