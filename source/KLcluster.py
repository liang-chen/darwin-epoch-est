
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.cluster import KMeans
from myClass import mGaussian
from math import log, sqrt
import random

def get_kl_dist(list_p,list_q):
    return sum([p*log(p/q,2) for p,q in zip(list_p, list_q)])

def get_mean_cols(matrix):
    s = []
    for i in xrange(len(matrix)):
        if len(s) == 0:
            s = matrix[i]
        else:
            s = [x+y for x,y in zip(s, matrix[i])]
    if len(s):
        s = [x/len(matrix) for x in s]

    return s

def KL_pair_trivial(KL_matrix):
    pList = []
    for i in xrange(len(KL_matrix)):
        row = KL_matrix[i,:]

        local_kl = i + random.random()/10 
        global_kl = i + random.random()/10
        pList.append([local_kl, global_kl])
    pairs = np.array(pList)
    return pairs

def KL_pair(KL_matrix, Topics_matrix):
    pList = []
    global_kl = []
    
    if(len(Topics_matrix) == 0):
        return np.array([])
    
    past = [1.0/ len(Topics_matrix[0])]*len(Topics_matrix[0])
    for i in xrange(len(Topics_matrix)):
        cur = Topics_matrix[i]
        global_kl.append(get_kl_dist(cur, past))
        past = get_mean_cols(Topics_matrix[0:i+1])
    
    for i in xrange(len(KL_matrix)):
        row = KL_matrix[i,:]
        if(i == 0):
            local_kl = 0
        #global_kl = 0
        else:
            local_kl = row[i-1]
        #global_kl = sum(row[:i])/i
        pList.append([local_kl, global_kl[i]])
    pairs = np.array(pList)
    return pairs

def KMC(pairs, n): #kmeans clustering on kl distance pairs
    k_means = KMeans(init='k-means++', n_clusters=n, n_init=10)
    k_means.fit(pairs)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FFD700', '#00FF7F', '#B22222', '#D02090', '#EED5B7']

    # KMeans Plot
    #fig = plt.figure(figsize=(8, 5))
    #ax = fig.add_subplot(1,1,1)
    #for k, col in zip(range(n), colors):
    #   my_members = k_means_labels == k
    #   cluster_center = k_means_cluster_centers[k]
    #   ax.plot(pairs[my_members, 0], pairs[my_members, 1], 'w', markerfacecolor=col, marker='.')
    #   ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    #   ax.set_xlabel('local suprise')
    #   ax.set_ylabel('global suprise')
    
    #plt.show()
    return k_means

def learn_KL_gaussian(k_means, pairs):
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    n = k_means.n_clusters
    kGaussians = []
    for k in range(n):
        my_members = k_means_labels == k
        my_center = sum(pairs[my_members,:]) / len(pairs[my_members,:])#k_means_cluster_centers[k]
        my_data = pairs[my_members,:]
        my_cov = np.cov(my_data, rowvar = 0)
        kGaussians.append(mGaussian(my_center, my_cov))
    
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    # KMeans Plot
#    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FFD700', '#00FF7F', '#B22222', '#D02090', '#EED5B7']
#    fig = plt.figure(figsize=(8, 5))
#    ax = fig.add_subplot(1,1,1)
#    for k, col in zip(range(n), colors):
#        my_members = k_means_labels == k
#        cluster_center = k_means_cluster_centers[k]
#        ax.plot(pairs[my_members, 0], pairs[my_members, 1], 'w', markerfacecolor=col, marker='.')
#        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
#
#    for k in range(n):
#        my_members = k_means_labels == k
#        my_center = sum(pairs[my_members,:]) / len(pairs[my_members,:])#k_means_cluster_centers[k]
#        my_data = pairs[my_members,:]
#        
#        my_data = np.sort(my_data, 0)
#        my_cov = np.cov(my_data, rowvar = 0)
#        
#        #print np.array(pairs[1:20,0]), np.array(pairs[1:20,1])
#        X, Y = np.meshgrid(np.array(my_data[:,0] ), np.array(my_data[:,1]) )
#        Z = mlab.bivariate_normal(X, Y, sqrt(my_cov[0,0]), sqrt(my_cov[1,1]), my_center[0], my_center[1], my_cov[0,1]/sqrt(my_cov[0,0])/sqrt(my_cov[1,1]))
#        CS = plt.contour(X,Y,Z,6)
#
#
#    ax.set_xlabel('local suprise')
#    ax.set_ylabel('global suprise')
#    plt.show()

    return kGaussians

def learn_from_KL(KL_matrix, Topics_matrix, n):
    pairs = KL_pair(KL_matrix, Topics_matrix)
    k_means_list = []

    for i in xrange(2, n+1):
        k_means_list.append(KMC(pairs, i))

    #k_means_t = KMC(pairs_t, n)
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(pairs[:, 0], pairs[:, 1], 'w', markerfacecolor='#000000', marker='.')
    ax.set_xlabel('local suprise')
    ax.set_ylabel('global suprise')
    plt.show()
    
    kGaussians = []
    for k_means in k_means_list:
        kGaussians = kGaussians + learn_KL_gaussian(k_means, pairs)
    return pairs, np.array(kGaussians)
