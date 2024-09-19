import random
import numpy as np
from scipy.stats import truncnorm


class RSU:
    def __init__(self, position, capacity, distance_from_bs) -> None:
        self.position = position
        self.distance_from_bs = distance_from_bs
        self.capacity = capacity

    def __repr__(self) -> str:
        return f"RSU at {self.position}, distance from BS: {self.distance_from_bs}"


class Vehicle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

    def __repr__(self) -> str:
        return f"Vehicle at {self.position}, velocity: {self.velocity}"


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
            self.vehicles.append(Vehicle(0, self.truncated_gaussian()))

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

        # remove vehicles that have left the road
        self.vehicles = [
            vehicle for vehicle in self.vehicles if vehicle.position <= self.road_length
        ]

        # update vehicle velocity
        for vehicle in self.vehicles:
            vehicle.velocity = self.truncated_gaussian()

        # add new vehicles
        for _ in range(int(self.poisson_event())):
            self.vehicles.append(Vehicle(0, self.truncated_gaussian()))

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
        time_step=1,
    )

    for _ in range(1000):
        print(f"Time: {env.current_time}")
        print(f"Number of vehicles: {len(env.vehicles)}")
        for idx, vehicle in enumerate(env.vehicles):
            print(f"Vehicle {idx} at {vehicle.position}, velocity: {vehicle.velocity}")
        env.small_step()
