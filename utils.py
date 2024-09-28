import numpy as np
import torch
import copy


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def cal_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cal_distance_matrix(vehicles, rsus):
    distance_matrix = np.zeros((len(vehicles), len(rsus)))
    for i, vehicle in enumerate(vehicles):
        for j, rsu in enumerate(rsus):
            distance_matrix[i][j] = cal_distance(vehicle.position, rsu.position)
    return distance_matrix
