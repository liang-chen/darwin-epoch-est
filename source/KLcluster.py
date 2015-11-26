
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def KL_pair(KL_matrix):
    pList = []
    for i in xrange(len(KL_matrix)):
        row = KL_matrix[i,:]

        if(i == 0):
            local_kl = 0
            global_kl = 0
        else:
            local_kl = row[i-1]
            global_kl = sum(row[:i])/i
        pList.append([local_kl, global_kl])
    pairs = np.array(pList)
    print pairs
    return pairs

def KMC(pairs, n): #kmeans clustering on kl distance pairs
    k_means = KMeans(init='k-means++', n_clusters=n, n_init=10)
    k_means.fit(pairs)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FFD700', '00FF7F', 'B22222', 'D02090', 'EED5B7']

    # KMeans
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1,1,1)
    for k, col in zip(range(n), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(pairs[my_members, 0], pairs[my_members, 1], 'w', markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    plt.show()
