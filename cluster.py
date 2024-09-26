import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def clustering(clients_num, flattened_weights):

    if clients_num == None:
        silhouette_score_lst = []
        for n_clusters in range(2, clients_num):
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(flattened_weights)
            print("cluster_labels", cluster_labels)

            # tinh toan he so silhoutte cho tung diem du lieu
            silhouette_avg = silhouette_score(flattened_weights, cluster_labels)

            silhouette_score_lst.append(silhouette_avg)

        # tim gia tri silhouette_avg lon nhat va so luong cum tuong ung
        max_silhouette_avg = max(silhouette_score_lst)
        best_n_clusters = silhouette_score_lst.index(max_silhouette_avg) + 2
        kmeans = KMeans(n_clusters=best_n_clusters)
        cluster_labels = kmeans.fit_predict(flattened_weights)
    else:
        best_n_clusters = clients_num
        kmeans = KMeans(n_clusters=best_n_clusters)
        cluster_labels = kmeans.fit_predict(flattened_weights)

    clusters_veh = []
    clusters_veh = [
        [index for index, value in enumerate(cluster_labels) if value == sublist_index]
        for sublist_index in set(cluster_labels)
    ]

    return clusters_veh, best_n_clusters
