import copy
import random
import numpy as np
from scipy.stats import truncnorm
import torch
from dataset import load_dataset
import pandas as pd
from model import AutoEncoder


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
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        input = torch.tensor(self.create_ratings_matrix()).float()

        patience = 10
        best_loss = float("inf")
        epochs_no_improve = 0

        for _ in range(1000):
            optimizer.zero_grad()
            output = self.model(input)
            loss = criterion(output, input)
            loss.backward()
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        return output

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)


class Environment:
    def __init__(
        self,
        lambda_poisson,
        min_velocity,
        max_velocity,
        std_velocity,
        road_length,
        rsu_coverage,
        rsu_capacity,
        num_rsu,
        num_vehicles,
        time_step=1,
        rsu_highway_distance=10,
        bs_highway_distance=10,
    ) -> None:

        assert min_velocity <= max_velocity and min_velocity >= 0, "Invalid velocity"
        assert num_rsu * rsu_coverage <= road_length, "Invalid RSU configuration"

        # Simulation parameters
        self.road_length = road_length
        self.time_step = time_step
        self.current_time = 0
        self.content_library = ContentLibrary("./data/ml-100k/")
        self.global_model = self.create_model()

        # RSU
        self.rsu_coverage = rsu_coverage
        self.rsu_capacity = rsu_capacity
        self.num_rsu = num_rsu

        # BS
        self.bs_position = -2000

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
            self.rsu.append(RSU(rsu_position, self.rsu_capacity, distance_from_bs))

        # Vehicle initialization
        self.vehicles = []
        positions = self.uniform_distribution(num_vehicles, road_length)

        for i in range(num_vehicles):
            self.add_vehicle(positions[i])

    def create_model(self):
        return AutoEncoder(self.content_library.max_item_id + 1, 512)

    def truncated_gaussian(self):
        a, b = (self.min_velocity - self.mean_velocity) / self.std_velocity, (
            self.max_velocity - self.mean_velocity
        ) / self.std_velocity
        return truncnorm.rvs(a, b, loc=self.mean_velocity, scale=self.std_velocity)

    # def poisson_event(self):
    #     return random.random() < (1 - np.exp(-self.lambda_poisson * self.current_time))

    def add_vehicle(self, position):
        user_id = self.content_library.get_user()
        user_info = self.content_library.load_user_info(user_id)
        user_data = self.content_library.load_ratings(user_id)
        self.vehicles.append(
            Vehicle(
                position,
                self.truncated_gaussian(),
                user_id,
                user_info,
                user_data,
                self.create_model(),
            )
        )

    def uniform_distribution(self, n, L):
        points = np.random.uniform(0, L, n)
        return points

    def small_step(self):
        # update vehicle positions
        for vehicle in self.vehicles:
            vehicle.update_position(
                vehicle.position + vehicle.velocity * self.time_step
            )

        # remove vehicles that have left the road and return the user id to the content library
        for vehicle in self.vehicles.copy():
            if vehicle.position > self.road_length:
                # remove vehicle from the list

                self.content_library.return_user(vehicle.user_id)
                self.vehicles.remove(vehicle)

                # then add drop new vehicle
                self.add_vehicle(0)

        # update vehicle velocity
        for vehicle in self.vehicles:
            vehicle.update_velocity(self.truncated_gaussian())

        # update time
        self.current_time += self.time_step


if __name__ == "__main__":
    env = Environment(
        lambda_poisson=0.05,
        min_velocity=5,
        max_velocity=10,
        std_velocity=2.5,
        road_length=2000,
        rsu_coverage=400,
        rsu_capacity=20,
        num_rsu=5,
        num_vehicles=100,
        time_step=10,
    )

    for _ in range(1000):
        print(f"Time: {env.current_time}")
        print(f"Number of vehicles: {len(env.vehicles)}")
        for idx, vehicle in enumerate(env.vehicles):
            print(vehicle)
        env.small_step()

        if len(env.vehicles) == 0:
            continue
        else:
            env.vehicles[0].local_update()
            weights = env.vehicles[0].get_weights()
