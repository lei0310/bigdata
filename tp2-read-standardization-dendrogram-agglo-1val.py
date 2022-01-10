# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:28:40 2021

@author: huguet
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features

# Note 1 : 
# dans les jeux de données considérés : 2 features (dimension 2 seulement)
# t =np.array([[1,2], [3,4], [5,6], [7,8]]) 
#
# Note 2 : 
# le jeu de données contient aussi un numéro de cluster pour chaque point
# --> IGNORER CETTE INFORMATION ....
#    2d-4c-no9.arff   xclara.arff

path = './new-data/'
databrut = arff.loadarff(open(path+"spiral.arff", 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])


########################################################################
# Preprocessing: standardization of data
########################################################################

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(datanp)

data_scaled = scaler.transform(datanp)

import scipy.cluster.hierarchy as shc

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
print("-----------------------------------------------------------")
print("Appel Aglo Clustering 'complete' pour une valeur de k fixée")
k=4
model_scaled = cluster.KMeans(n_clusters=k, init='k-means++')
model_scaled.fit(data_scaled)
#cluster.fit_predict(X)

labels_scaled = model_scaled.labels_
#print(tps4)

plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
plt.title("Données (std) après clustering")
plt.show()
#print("nb clusters =",k,", runtime = ", round((tps4 - tps3)*1000,2),"ms")
#print("labels", labels)


# Méthode du coude
tps3 = time.time()
inert = model_scaled.inertia_
print("Inertie : ", inert)
sum_err=[]
for i in range(2,10):
  km=cluster.KMeans(n_clusters=i)
  km.fit(datanp)
  sum_err.append(km.inertia_)

tps4 = time.time()
print("Temps exécution méthode du coude", round((tps4 - tps3)*1000,2),"ms")

plt.plot(range(2,10),sum_err)
plt.xlabel("Nombre de clusters k")
plt.ylabel("Somme des carré des distances")
plt.title("Méthode du coude sur le jeu de données spiral.arff")
plt.show()

#Coefficient de silhouette

tps5 = time.time()
sum_err=[]
for i in range(2,10):
    km = cluster.KMeans(n_clusters=i)
    km.fit(datanp)
    sum_err.append(metrics.silhouette_score(datanp, km.labels_))

tps6 = time.time()
print("Temps exécution méthode de la silhouette", round((tps6 - tps5)*1000,2),"ms")

plt.plot(range(2,10),sum_err)
plt.xlabel("Nombre de clusters k")
plt.ylabel("Coefficient de silhouette")
plt.title("Calcul du coefficient de silhouette sur le jeu de données spiral.arff")
plt.show()




