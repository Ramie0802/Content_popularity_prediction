import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def clustering(args, flattened_weights):
    
    if args.clusters_num == None:
        silhouette_score_lst = []
        for n_clusters in range(2, args.clients_num):
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(flattened_weights)
            print('cluster_labels', cluster_labels)

            # tinh toan he so silhoutte cho tung diem du lieu
            silhouette_avg = silhouette_score(flattened_weights, cluster_labels)            

            silhouette_score_lst.append(silhouette_avg)   
        
        # tim gia tri silhouette_avg lon nhat va so luong cum tuong ung
        max_silhouette_avg = max(silhouette_score_lst)
        best_n_clusters = silhouette_score_lst.index(max_silhouette_avg) + 2
        kmeans = KMeans(n_clusters=best_n_clusters)
        cluster_labels = kmeans.fit_predict(flattened_weights)
    else:
        best_n_clusters = args.clusters_num
        kmeans = KMeans(n_clusters=best_n_clusters)
        cluster_labels = kmeans.fit_predict(flattened_weights)
    

    print(f"\n============best cluster: {best_n_clusters}=============\n")

   
    print("\n==============cluster_labels=============\n")
    print(cluster_labels)    

    clusters_veh = []
    clusters_veh = [[index for index, value in enumerate(cluster_labels) if value == sublist_index] for sublist_index in set(cluster_labels)]
    print(clusters_veh)   
        


    return clusters_veh, best_n_clusters


def choose_clusterhead(veh_capacity, clusters_veh): 
    
    cluster_head_dct = {
        'cluster_idx':[],
        'cluster_head_idx': [],
        'capacity': []
    }

    for cluster in clusters_veh:
        max_capacity = 0
        cluster_head = None

        for veh in cluster:
            if veh_capacity[veh] > max_capacity:
                max_capacity = veh_capacity[veh]
                cluster_head = veh
        cluster_head_dct['cluster_idx'].append(cluster)
        cluster_head_dct['cluster_head_idx'].append(cluster_head)
        cluster_head_dct['capacity'].append(max_capacity)
    
    return cluster_head_dct

