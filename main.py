from communication import Communication
from environment import Environment
import concurrent.futures
from utils import cal_distance_matrix, average_weights
import numpy as np
from cluster import clustering


min_velocity = 5
max_velocity = 10
std_velocity = 2.5

road_length = 2000
road_width = 10

rsu_coverage = 400
rsu_capacity = 20

num_rsu = 5
num_vehicles = 40

time_step = 1
num_clusters = 5


env = Environment(
    min_velocity=min_velocity,
    max_velocity=max_velocity,
    std_velocity=std_velocity,
    road_length=road_length,
    road_width=road_width,
    rsu_coverage=rsu_coverage,
    rsu_capacity=rsu_capacity,
    num_rsu=num_rsu,
    num_vehicles=num_vehicles,
    time_step=time_step,
)

for _ in range(1):

    # load mobility status
    distance_matrix = cal_distance_matrix(env.vehicles, env.rsu)

    # load communication status
    env.communication = Communication(distance_matrix)

    # check coverage status
    coverage = {k: [] for k in range(num_rsu)}
    for i in range(num_vehicles):
        for j in range(num_rsu):
            if env.communication.distance_matrix[i][j] < rsu_coverage // 2:
                coverage[j].append(i)

    # select vehicles
    if True:  # modify this condition
        selected_vehicles = env.vehicles

    # perform local training

    def local_update(vehicle):
        vehicle.local_update()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(local_update, selected_vehicles)

    # flatten all weights and perform clustering
    weights = [vehicle.get_weights() for vehicle in selected_vehicles]

    flattened_weights = [
        np.concatenate([np.array(v).flatten() for v in w.values()]) for w in weights
    ]

    clusters = []

    for i in range(num_rsu):
        if len(coverage[i]) >= num_clusters:
            cluster, _ = clustering(
                num_clusters, [flattened_weights[j] for j in coverage[i]]
            )
        else:
            cluster, _ = clustering(1, [flattened_weights[j] for j in coverage[i]])

        # get the original indices
        cluster = [[coverage[i][j] for j in c] for c in cluster]
        for c in cluster:
            clusters.append(c)

    # aggregate weights
    for cluster in clusters:
        cluster_weights = average_weights([weights[i] for i in cluster])

        for idx in cluster:
            env.vehicles[idx].set_weights(cluster_weights)

    # perform prediction
    for r in range(num_rsu):
        predictions = []
        for i in coverage[r]:
            predictions.append(env.vehicles[i].predict().detach().numpy())

        # aggregate predictions
        popularity = np.mean(predictions, axis=0)

        # cache replacement
        cache = np.argsort(popularity)[::-1][: env.rsu[r].capacity]

        env.rsu[r].cache = cache

    # env
    for rsu in env.rsu:
        print(rsu.cache)

    env.small_step()
