import numpy as np
from scipy.stats import truncnorm
from library import ContentLibrary
from mobility import Vehicle
from model import AutoEncoder
from utils import cal_distance


class RSU:
    def __init__(self, position, capacity, distance_from_bs) -> None:
        self.position = position
        self.distance_from_bs = distance_from_bs
        self.capacity = capacity

    def __repr__(self) -> str:
        return f"id: {self.position}, capacity: {self.capacity}"


class Environment:
    def __init__(
        self,
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
        bs_position=(10, -2000),
        rsu_highway_distance=1,
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
        self.rsu_highway_distance = rsu_highway_distance

        # BS
        self.bs = RSU(bs_position, 100000, 0)

        # Vehicle parameters
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.std_velocity = std_velocity
        self.mean_velocity = (min_velocity + max_velocity) / 2
        self.num_vehicles = num_vehicles

        # RSU/BS placement
        self.rsu = []
        for i in range(num_rsu):
            rsu_position = (
                (i + 1) * rsu_coverage - rsu_coverage / 2,
                self.road_width + self.rsu_highway_distance,
            )
            distance_from_bs = cal_distance(self.bs.position, rsu_position)
            self.rsu.append(
                RSU(
                    rsu_position,
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
