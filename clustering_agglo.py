# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:57:02 2022

@author: lmene
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.cluster.hierarchy as shc

from sklearn import preprocessing
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


path = './artificial/'
databrut = arff.loadarff(open(path+"banana.arff", 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

scaler = preprocessing.StandardScaler().fit(datanp)

data_scaled = scaler.transform(datanp)

print("-------------------------------------------")
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

# print("-----------------------------------------")
# print("Dendrogramme 'complete' données standardisées")

# distance = shc.linkage(data_scaled, 'complete')


# plt.figure(figsize=(12, 12))
# shc.dendrogram(distance,
#             orientation='top',
#             distance_sort='descending',
#             show_leaf_counts=False)
# plt.show()

# Run clustering method for a given number of clusters
sum_err=[] #initialization du vecteur
print("-----------------------------------------------------------")
print("Appel Aglo Clustering 'ward' pour une valeur de k fixée")
tps3 = time.time()
k=3
model_scaled = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
model_scaled.fit(data_scaled)
#cluster.fit_predict(X)

tps4 = time.time()
labels_scaled = model_scaled.labels_

plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
plt.title("Clustering ward")
plt.show()
print("nb clusters =",k,", runtime = ", round((tps4 - tps3)*1000,2),"ms")
silh = metrics.silhouette_score(datanp, labels_scaled, metric='euclidean')
sum_err.append(silh)
#print("labels", labels)

# Run clustering method for a given number of clusters
print("-----------------------------------------------------------")
print("Appel Aglo Clustering 'complete' pour une valeur de k fixée")
tps3 = time.time()
k=3
model_scaled = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='complete')
model_scaled.fit(data_scaled)
#cluster.fit_predict(X)

tps4 = time.time()
labels_scaled = model_scaled.labels_

plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
plt.title("Clustering complete")
plt.show()
print("nb clusters =",k,", runtime = ", round((tps4 - tps3)*1000,2),"ms")
silh = metrics.silhouette_score(datanp, labels_scaled, metric='euclidean')
sum_err.append(silh)
#print("labels", labels)

# Run clustering method for a given number of clusters
print("-----------------------------------------------------------")
print("Appel Aglo Clustering 'single' pour une valeur de k fixée")
tps3 = time.time()
k=3
model_scaled = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='single')
model_scaled.fit(data_scaled)
#cluster.fit_predict(X)

tps4 = time.time()
labels_scaled = model_scaled.labels_

plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
plt.title("Clustering single")
plt.show()
print("nb clusters =",k,", runtime = ", round((tps4 - tps3)*1000,2),"ms")
#print("labels", labels)
silh = metrics.silhouette_score(datanp, labels_scaled, metric='euclidean')
sum_err.append(silh)

# Run clustering method for a given number of clusters
print("-----------------------------------------------------------")
print("Appel Aglo Clustering 'average' pour une valeur de k fixée")
tps3 = time.time()
k=3
model_scaled = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average')
model_scaled.fit(data_scaled)
#cluster.fit_predict(X)

tps4 = time.time()
labels_scaled = model_scaled.labels_

plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
plt.title("Clustering average")
plt.show()
print("nb clusters =",k,", runtime = ", round((tps4 - tps3)*1000,2),"ms")
#print("labels", labels)
silh = metrics.silhouette_score(datanp, labels_scaled, metric='euclidean')
sum_err.append(silh)

# Calcul de l'inertie des différentes méthodes 
plt.plot(["ward","complete","single","average"],sum_err)
plt.title('Evaluation de l inertie des méthodes de clustering agglomératifs')
plt.ylabel('Inertie')
plt.show()

# Calcul des coeff de silhouette des différentes méthodes 

sum_err_21=[]
sum_err_22=[]
sum_err_23=[]
sum_err_24=[]

link=['ward','complete','single','average']
for j in link :
  for i in range (2,10):
    sil=cluster.AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage=j)
    sil.fit(data_scaled)
    if (j=="ward"):
      sum_err_21.append(metrics.silhouette_score(datanp, sil.labels_, metric='euclidean'))
    if (j=="complete"):
      sum_err_22.append(metrics.silhouette_score(datanp, sil.labels_, metric='euclidean'))
    if (j=="single"):
      sum_err_23.append(metrics.silhouette_score(datanp, sil.labels_, metric='euclidean'))
    if (j=="average"):
      sum_err_24.append(metrics.silhouette_score(datanp, sil.labels_, metric='euclidean'))
      
plt.figure(1)
plt.plot(range(2,10),sum_err_21,label='ward')
plt.plot(range(2,10),sum_err_22,label='complete')
plt.plot(range(2,10),sum_err_23,label='single')
plt.plot(range(2,10),sum_err_24,label='average')
plt.legend()
plt.title('Evaluation du coefficient de silhouette des méthodes de clustering agglomératifs')
plt.show()

##############################  DEUXIEME JEU DE DONNEES ########################################################


