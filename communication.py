import numpy as np


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
        self.current_shadowing = self.compute_shadowing()
        self.current_path_loss = self.compute_path_loss()

    def compute_path_loss(self):
        path_loss_values = np.zeros((self.num_vehicle, self.num_rsu))

        for i in range(self.num_vehicle):
            for j in range(self.num_rsu):
                # Calculate distance between vehicle and RSU
                distance_value = self.distance_matrix[i][j]

                # Calculate path loss using the log-distance model
                path_loss = 128.1 + 37.6 * np.log10(distance_value)

                path_loss_values[i][j] = path_loss

        return path_loss_values

    def get_shadowing(self):
        return self.current_shadowing

    def compute_shadowing(self):
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

        # get path loss in dB
        path_loss_dB = self.current_path_loss[i][j]

        # get shadowing in dB
        shadowing_dB = self.current_shadowing[i][j]

        # Total attenuation (Path loss + Shadowing) in dB
        total_loss_dB = path_loss_dB + shadowing_dB

        # Convert total loss from dB to linear scale (channel gain in Watts)
        channel_gain_linear = 10 ** (-total_loss_dB / 10)

        # Convert transmission power and noise power from dBm to Watts
        transmission_power_watt = self.dBm_to_Watt(self.transmission_power_dBm)
        noise_power_watt = self.dBm_to_Watt(self.noise_power_dBm)

        # Calculate data rate using Shannon capacity formula
        data_rate = self.B_sub * np.log2(
            1 + (transmission_power_watt * channel_gain_linear) / (noise_power_watt)
        )

        return data_rate
