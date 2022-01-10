# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:05:00 2022

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
databrut = arff.loadarff(open(path+"sizes3.arff", 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

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

# Méthode DBSCAN pour un nombre donné de epsilon et min_samples

epsilon=0.15
min_pts=5
cl_pred = cluster.DBSCAN(eps=epsilon, min_samples=min_pts).fit_predict(data_scaled)

plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
plt.title("Clustering DBSCAN - Epilson=0.15 - Minpt=5")
plt.show()


n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
n_noise_ = list(cl_pred).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(radius=3)
nombre=neigh.fit(data_scaled)

liste_distance, liste_indice=nombre.kneighbors(data_scaled)
distance=[]

for i in range (len(liste_distance)):
  distance.append(liste_distance[i].mean())

plt.plot(range(0,len(distance)),sorted(distance))
plt.title('Nearest Neighbors : distance and associated indices')
plt.ylabel('epsilon')
# #print(distance)


# eps_range = [0.05, 0.06, 0.07]
# min_pts_max_range = 50
# tab_silh_0 = []
# tab_silh_1 = []
# tab_silh_2 = []

# for j in eps_range:
#     for i in range(1, min_pts_max_range):
#         cl_pred = cluster.DBSCAN(eps=j, min_samples=i).fit_predict(data_scaled)
#         n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
#         if(n_clusters_>1):
#           if(j==0.05) : {tab_silh_0.append(metrics.silhouette_score(data_scaled, cl_pred, metric='euclidean'))}
#           elif(j==0.06): {tab_silh_1.append(metrics.silhouette_score(data_scaled, cl_pred, metric='euclidean'))}
#           elif(j==0.07): {tab_silh_2.append(metrics.silhouette_score(data_scaled, cl_pred, metric='euclidean'))}

# plt.plot(range(0,len(tab_silh_0)), tab_silh_0, label='0.05')
# plt.plot(range(0,len(tab_silh_1)), tab_silh_1, label='0.06')
# plt.plot(range(0,len(tab_silh_2)), tab_silh_2, label='0.07')
# plt.legend()