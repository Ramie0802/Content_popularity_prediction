import copy
import random
import numpy as np
from scipy.stats import truncnorm
import torch
from dataset import load_dataset
import pandas as pd
from model import AutoEncoder
import matplotlib.pyplot as plt


def cal_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cal_distance_matrix(vehicles, rsus):
    distance_matrix = np.zeros((len(vehicles), len(rsus)))
    for i, vehicle in enumerate(vehicles):
        for j, rsu in enumerate(rsus):
            distance_matrix[i][j] = cal_distance(vehicle.position, rsu.position)
    return distance_matrix


class ContentLibrary:
    def __init__(self, path):
        self.dataset = load_dataset(path)
        self.ratings = self.dataset["ratings"]
        self.users = self.dataset["users"]
        self.semantic_vecs = self.dataset["semantic_vec"]
        self.sparse_vecs = self.dataset["sparse_vec"]
        self.semantic_cosine = self.dataset["cosine_semantic"]
        self.sparse_cosine = self.dataset["cosine_sparse"]

        self.total_items = list(set(self.ratings["item_id"]))
        self.total_users = list(set(self.ratings["user_id"]))

        self.user_sorted = self.ratings["user_id"].value_counts()

        self.used_user_ids = []

        self.max_item_id = max(self.total_items)

    def load_ratings(self, user_id):
        ratings = self.ratings[self.ratings["user_id"] == user_id].copy()
        ratings.loc[:, "semantic_vec"] = ratings["item_id"].apply(
            lambda x: self.semantic_vecs[int(x)]
        )
        ratings.loc[:, "sparse_vec"] = ratings["item_id"].apply(
            lambda x: self.sparse_vecs[int(x)]
        )

        ratings = ratings.sort_values(by="timestamp", ascending=False)

        return {
            "contents": ratings["item_id"].values,
            "ratings": ratings["rating"].values,
            "semantic_vecs": ratings["semantic_vec"].values,
            "sparse_vecs": ratings["sparse_vec"].values,
            "max": self.max_item_id,
        }

    def load_user_info(self, user_id):
        return self.ratings[self.ratings["user_id"] == user_id]

    def get_user(self):
        user_id = random.choice(self.total_users)
        while user_id in self.used_user_ids:
            user_id = random.choice(self.total_users)
        self.used_user_ids.append(user_id)
        return user_id

    def return_user(self, user_id):
        self.used_user_ids.remove(user_id)


class RSU:
    def __init__(self, position, capacity, distance_from_bs) -> None:
        self.position = position
        self.distance_from_bs = distance_from_bs
        self.capacity = capacity

    def __repr__(self) -> str:
        return f"id: {self.position}, capacity: {self.capacity}"


class Vehicle:
    def __init__(self, position, velocity, user_id, info, data, model) -> None:
        self.user_id = user_id
        self.position = position
        self.velocity = velocity
        self.data = data
        self.info = info
        self.divider = int(len(data["contents"]) * 0.8)

        self.input_shape = self.data["max"] + 1

        # load the model architecture
        self.model = model

        # generate request from the test set
        self.current_request = self.generate_request()

    def __repr__(self) -> str:
        return (
            f"id: {self.user_id}, position: {self.position}, velocity: {self.velocity}"
        )

    def update_velocity(self, velocity):
        self.velocity = velocity

    def update_position(self, position):
        self.position = position

    def update_request(self):
        self.divider += 1
        self.current_request = self.generate_request()

    def create_ratings_matrix(self):
        matrix = []
        for i in range(self.data["max"] + 1):
            if i in self.data["contents"][: self.divider]:
                matrix.append(1)
            else:
                matrix.append(0)
        return np.array(matrix)

    def generate_request(self):
        return random.choice(self.data["contents"][self.divider :])

    def local_update(self):
        self.model.train()
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        input = torch.tensor(self.create_ratings_matrix()).float()

        patience = 10
        best_loss = float("inf")
        epochs_no_improve = 0
        best_weights = copy.deepcopy(self.model.state_dict())

        for _ in range(4):
            optimizer.zero_grad()
            output = self.model(input)
            loss = criterion(output, input)
            loss.backward()
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_no_improve = 0
                best_weights = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        self.model.load_state_dict(best_weights)
        return output

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)


class Communication:
    def __init__(self, distance_matrix):
        self.B = 20e6  # 20 MHz
        self.B_sub = 500e3  # 500 kHz
        self.fiber = 100e6  # Fiber bandwidth in bps
        self.shadow_std = 4  # Shadow fading standard deviation in dB
        self.transmission_power_dBm = 30  # Transmission power in dBm
        self.noise_power_dBm = -114  # Noise power in dBm
        self.decorrelation_distance = 50  # Decorrelation distance in meters
        self.distance_matrix = distance_matrix
        self.num_vehicle, self.num_rsu = distance_matrix.shape
        self.current_shadowing = self.get_shadowing()

    def compute_path_loss(self, distance):
        if distance <= 0:
            raise ValueError("Distance must be greater than zero.")

        return 128.1 + 37.6 * np.log10(distance * 1e-3)

    def get_shadowing(self):
        shadowing_values = np.zeros((self.num_vehicle, self.num_rsu))

        for i in range(self.num_vehicle):
            # Generate a base shadowing value for each vehicle
            base_shadowing = np.random.normal(0, self.shadow_std)

            for j in range(self.num_rsu):
                # Calculate the delta distance from the distance matrix
                delta_distance = self.distance_matrix[i][j]

                # Calculate shadowing using the decorrelation model
                shadowing_decay = np.exp(
                    -1 * (delta_distance / self.decorrelation_distance)
                )

                additional_shadowing = np.sqrt(
                    1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))
                ) * np.random.normal(0, 4)

                shadowing_values[i][j] = (
                    base_shadowing * shadowing_decay + additional_shadowing
                )

        for i in range(self.num_vehicle):
            for j in range(self.num_rsu):
                if shadowing_values[i][j] is None:
                    print(i, j)

        return shadowing_values

    def dBm_to_Watt(self, dBm):
        # Convert dBm to Watts
        return 10 ** ((dBm - 30) / 10)

    def calculate_V2R_data_rate(self, vi, rj, i, j):
        # Compute distance between vehicle and RSU
        distance_value = self.distance_matrix[i][j]

        # Compute path loss in dB
        path_loss_dB = self.compute_path_loss(distance_value)

        shadowing = self.current_shadowing[i][j]

        # Get shadowing in dB
        # shadowing_dB = 10 * np.log10(self.current_shadowing[i][j])
        shadowing_dB = shadowing

        # Total attenuation (Path loss + Shadowing) in dB
        total_loss_dB = path_loss_dB + shadowing_dB

        # Convert total loss from dB to linear scale (channel gain in Watts)
        channel_gain_linear = 10 ** (-total_loss_dB / 10)

        # Convert transmission power and noise power from dBm to Watts
        transmission_power_watt = self.dBm_to_Watt(self.transmission_power_dBm)
        noise_power_watt = self.dBm_to_Watt(self.noise_power_dBm)

        print(
            f"Distance: {distance_value}, Path Loss: {path_loss_dB}, Shadowing: {shadowing}, Channel Gain: {channel_gain_linear}, Transmission Power: {transmission_power_watt}, Noise Power: {noise_power_watt}"
        )

        # Calculate data rate using Shannon capacity formula
        data_rate = self.B_sub * np.log2(
            1 + (transmission_power_watt * channel_gain_linear) / (noise_power_watt)
        )

        return data_rate


class Environment:
    def __init__(
        self,
        lambda_poisson,
        min_velocity,
        max_velocity,
        std_velocity,
        road_length,
        road_width,
        rsu_coverage,
        rsu_capacity,
        num_rsu,
        num_vehicles,
        time_step=1,
        rsu_highway_distance=1,
        bs_highway_distance=10,
    ) -> None:

        assert min_velocity <= max_velocity and min_velocity >= 0, "Invalid velocity"
        assert num_rsu * rsu_coverage <= road_length, "Invalid RSU configuration"

        # Simulation parameters
        self.road_length = road_length
        self.road_width = road_width
        self.time_step = time_step
        self.current_time = 0
        self.content_library = ContentLibrary("./data/ml-100k/")
        self.global_model = self.init_model()

        # RSU
        self.rsu_coverage = rsu_coverage
        self.rsu_capacity = rsu_capacity
        self.num_rsu = num_rsu

        # BS
        self.bs = RSU((10, -2000), 100000, 0)

        # Vehicle parameters
        self.lambda_poisson = lambda_poisson
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.std_velocity = std_velocity
        self.mean_velocity = (min_velocity + max_velocity) / 2
        self.num_vehicles = num_vehicles

        # RSU/BS placement
        self.rsu = []
        for i in range(num_rsu):
            rsu_position = (i + 1) * rsu_coverage - rsu_coverage / 2
            distance_from_bs = abs(rsu_position - self.bs_position)
            self.rsu.append(
                RSU(
                    (road_width + rsu_highway_distance, rsu_position),
                    self.rsu_capacity,
                    distance_from_bs,
                )
            )

        # Vehicle initialization
        self.vehicles = []
        x_positions, y_positions = self.poisson_process_on_road(
            num_vehicles, road_length, road_width
        )

        for i in range(num_vehicles):
            self.add_vehicle(x_positions[i], y_positions[i])

    def init_model(self):
        return AutoEncoder(self.content_library.max_item_id + 1, 512)

    def truncated_gaussian(self):
        a, b = (self.min_velocity - self.mean_velocity) / self.std_velocity, (
            self.max_velocity - self.mean_velocity
        ) / self.std_velocity
        return truncnorm.rvs(a, b, loc=self.mean_velocity, scale=self.std_velocity)

    def add_vehicle(self, x_position, y_position):
        user_id = self.content_library.get_user()
        user_info = self.content_library.load_user_info(user_id)
        user_data = self.content_library.load_ratings(user_id)
        self.vehicles.append(
            Vehicle(
                (x_position, y_position),
                self.truncated_gaussian(),
                user_id,
                user_info,
                user_data,
                self.init_model(),
            )
        )

    def poisson_process_on_road(self, n, length, width):
        # Generate n points with random x (length) and y (width) positions
        x_positions = np.random.uniform(0, length, n)
        y_positions = np.random.uniform(0, width, n)
        return x_positions, y_positions

    def small_step(self):
        # update vehicle positions
        for vehicle in self.vehicles:
            new_x_position = vehicle.position[0] + vehicle.velocity * self.time_step
            vehicle.update_position((new_x_position, vehicle.position[1]))

        # remove vehicles that have left the road and return the user id to the content library
        for vehicle in self.vehicles.copy():
            if vehicle.position[0] > self.road_length:
                # remove vehicle from the list
                self.content_library.return_user(vehicle.user_id)
                self.vehicles.remove(vehicle)

                x_positions, y_positions = self.poisson_process_on_road(
                    1, self.road_length, self.road_width
                )
                self.add_vehicle(0, y_positions[0])

        # update vehicle velocity
        for vehicle in self.vehicles:
            vehicle.update_velocity(self.truncated_gaussian())

        # update time
        self.current_time += self.time_step


if __name__ == "__main__":

    width = 10
    length = 2000

    env = Environment(
        lambda_poisson=0.05,
        min_velocity=5,
        max_velocity=10,
        std_velocity=2.5,
        road_length=length,
        road_width=width,
        rsu_coverage=400,
        rsu_capacity=20,
        num_rsu=5,
        num_vehicles=20,
        time_step=10,
    )
    for _ in range(10):
        print(f"Time: {env.current_time}")
        print(f"Number of vehicles: {len(env.vehicles)}")
        rsu = env.rsu[0]
        distance_matrix = cal_distance_matrix(env.vehicles, env.rsu)
        env.communication = Communication(distance_matrix)
        for i, vehicle in enumerate(env.vehicles):
            if distance_matrix[i, 0] < 400:
                data_rate = env.communication.calculate_V2R_data_rate(
                    vehicle, rsu, i, 0
                )
                print("Distance: ", distance_matrix[i, 0], "Data Rate: ", data_rate)

#     plt.figure(figsize=(12, 3))

#     for _ in range(1000):
#         print(f"Time: {env.current_time}")
#         print(f"Number of vehicles: {len(env.vehicles)}")
#         # for idx, vehicle in enumerate(env.vehicles):
#         #     print(vehicle)

#         if len(env.vehicles) == 0:
#             continue
#         else:
#             env.vehicles[0].local_update()
#             weights = env.vehicles[0].get_weights()

#         plt.clf()
#         x_positions = [vehicle.position[0] for vehicle in env.vehicles]
#         y_positions = [vehicle.position[1] for vehicle in env.vehicles]

#         plt.hlines([0, width], 0, length, colors="gray", linestyles="solid")
#         plt.scatter(x_positions, y_positions, c="blue", s=100, label="Points")

#         plt.title(f"Iteration {_+1}, num vehicles: {len(env.vehicles)}")
#         plt.xlim(0, length)
#         plt.ylim(0 - 10, width + 10)
#         plt.xlabel("Position along the Road (m)")
#         plt.ylabel("Width of the Road (m)")

#         # Display the updated plot without creating a new figure
#         plt.pause(0.1)

#         env.small_step()

# plt.show()
