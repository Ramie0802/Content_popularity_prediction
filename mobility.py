import copy
import numpy as np
import torch


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
        return np.random.choice(self.data["contents"][self.divider :])

    def predict(self):
        self.model.eval()
        input = torch.tensor(self.create_ratings_matrix()).float()
        return self.model(input)

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
