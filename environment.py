import random
import numpy as np
from scipy.stats import truncnorm
from dataset import load_dataset
import pandas as pd


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

    def load_ratings(self, user_id):
        ratings = self.ratings[self.ratings["user_id"] == user_id].copy()
        ratings.loc[:, "semantic_vec"] = ratings["item_id"].apply(
            lambda x: self.semantic_vecs[int(x)]
        )
        ratings.loc[:, "sparse_vec"] = ratings["item_id"].apply(
            lambda x: self.sparse_vecs[int(x)]
        )

        ratings = ratings.sort_values(by="timestamp", ascending=False)
        return ratings

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
        return f"RSU at {self.position}, distance from BS: {self.distance_from_bs}"


class Vehicle:
    def __init__(self, position, velocity, user_id, info, data) -> None:
        self.user_id = user_id
        self.position = position
        self.velocity = velocity
        self.data = data
        self.info = info

    def __repr__(self) -> str:
        return f"Vehicle {self.user_id} at {self.position}, velocity: {self.velocity}"


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
        time_step=1,
    ) -> None:

        assert min_velocity <= max_velocity and min_velocity >= 0, "Invalid velocity"
        assert num_rsu * rsu_coverage <= road_length, "Invalid RSU configuration"

        # Simulation parameters
        self.road_length = road_length
        self.time_step = time_step
        self.current_time = 0
        self.content_library = ContentLibrary("./data/ml-100k/")

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

        # RSU/BS placement
        self.rsu = []
        for i in range(num_rsu):
            rsu_position = (i + 1) * rsu_coverage - rsu_coverage / 2
            distance_from_bs = abs(rsu_position - self.bs_position)
            self.rsu.append(RSU(rsu_position, self.rsu_capacity, distance_from_bs))

        # Vehicle initialization
        self.vehicles = []

        for _ in range(int(self.poisson_event())):
            user_id = self.content_library.get_user()
            user_info = self.content_library.load_user_info(user_id)
            user_data = self.content_library.load_ratings(user_id)
            self.vehicles.append(
                Vehicle(0, self.truncated_gaussian(), user_id, user_info, user_data)
            )

    def truncated_gaussian(self):
        a, b = (self.min_velocity - self.mean_velocity) / self.std_velocity, (
            self.max_velocity - self.mean_velocity
        ) / self.std_velocity
        return truncnorm.rvs(a, b, loc=self.mean_velocity, scale=self.std_velocity)

    def poisson_event(self):
        return random.random() < (1 - np.exp(-self.lambda_poisson * self.current_time))

    def small_step(self):
        # update vehicle positions
        for vehicle in self.vehicles:
            vehicle.position += vehicle.velocity * self.time_step

        # remove vehicles that have left the road and return the user id to the content library
        for vehicle in self.vehicles.copy():
            if vehicle.position > self.road_length:
                self.content_library.return_user(vehicle.user_id)
                self.vehicles.remove(vehicle)

        # update vehicle velocity
        for vehicle in self.vehicles:
            vehicle.velocity = self.truncated_gaussian()

        # add new vehicles
        for _ in range(int(self.poisson_event())):
            user_id = self.content_library.get_user()
            user_info = self.content_library.load_user_info(user_id)
            user_data = self.content_library.load_ratings(user_id)
            self.vehicles.append(
                Vehicle(0, self.truncated_gaussian(), user_id, user_info, user_data)
            )

        # update time
        self.current_time += self.time_step


if __name__ == "__main__":
    env = Environment(
        lambda_poisson=0.1,
        min_velocity=5,
        max_velocity=10,
        std_velocity=2.5,
        road_length=2000,
        rsu_coverage=400,
        rsu_capacity=20,
        num_rsu=5,
        time_step=10,
    )

    for _ in range(1000):
        print(f"Time: {env.current_time}")
        print(f"Number of vehicles: {len(env.vehicles)}")
        for idx, vehicle in enumerate(env.vehicles):
            print(vehicle, len(env.content_library.used_user_ids))
        env.small_step()
