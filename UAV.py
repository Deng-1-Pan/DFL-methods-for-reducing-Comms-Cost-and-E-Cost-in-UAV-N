import matplotlib.pyplot as plt
import numpy as np
import random
import argparse, json
import datetime
import os
import logging

from sklearn.cluster import KMeans
from scipy import spatial


def uav_register(coords):
    Kmean = KMeans(n_clusters=2, init='k-means++',random_state=0)
    Kmean.fit(coords)
    
    cluster0 = coords[Kmean.fit_predict(coords) == 0] 
    cluster1 = coords[Kmean.fit_predict(coords) == 1]
    label_coords0 = sorted(set(np.nonzero(cluster0[:,None] == coords)[1]))
    label_coords1 = sorted(set(np.nonzero(cluster1[:,None] == coords)[1]))
    
    distance0,index0 = spatial.KDTree(coords).query(Kmean.cluster_centers_[0])
    distance1,index1 = spatial.KDTree(coords).query(Kmean.cluster_centers_[1]) 
   
    return index0, index1, label_coords0, label_coords1