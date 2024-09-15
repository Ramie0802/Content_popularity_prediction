import torch
from torch import nn
import pandas as pd
import numpy as np


from collections import Counter
from torch.utils.data import DataLoader, Dataset
import copy


class MovieDataset(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        row = self.dataset[self.idxs[item]]
        return torch.tensor(row, dtype=torch.float32)


# Federated branch
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.dataset = dataset[idxs]
        self.idxs = np.arange(0, len(self.dataset))
        self.local_bs = 50
        self.trainloader = DataLoader(
            MovieDataset(self.dataset, self.idxs),
            batch_size=self.local_bs,
            shuffle=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.learning_rate = 0.01
        self.criterion = nn.L1Loss().to(self.device)
        self.local_epochs = 1

    def update_weights(self, model, client_idx, global_round):

        # model.to(self.device)
        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for iter in range(self.local_epochs):
            batch_loss = []
            for batch_idx, data_sample in enumerate(self.trainloader):

                x_train = data_sample[:, :-1].to(self.device)
                y_train = data_sample[:, -1].to(self.device)
                # print("x_train/y_train")
                # print(x_train.shape, y_train.shape)

                optimizer.zero_grad()
                outputs = model(x_train)
                loss = self.criterion(outputs.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Round : {} | Vehicle : {} | Branch 1|Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            client_idx,
                            iter + 1,
                            batch_idx * len(x_train),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return (
            model.state_dict(),
            sum(epoch_loss) / len(epoch_loss),
            copy.deepcopy(model),
        )


# LSTM - semantic branch
class LocalUpdate_branch3(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.dataset = dataset[idxs]
        self.idxs = np.arange(0, len(self.dataset))
        self.local_bs = 50
        self.trainloader = DataLoader(
            MovieDataset(self.dataset, self.idxs),
            batch_size=self.local_bs,
            shuffle=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.learning_rate = 0.01
        self.criterion = nn.L1Loss().to(self.device)
        self.local_epochs = 1

    def update_weights(self, model, client_idx, global_round):

        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for iter in range(self.local_epochs):
            batch_loss = []
            for batch_idx, data_sample in enumerate(self.trainloader):

                x_train = data_sample[:, :-1].to(self.device)
                y_train = data_sample[:, -1].to(self.device)

                optimizer.zero_grad()
                outputs = model(x_train)

                loss = self.criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Round : {} | Vehicle : {} | Branch 3| Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            client_idx,
                            iter + 1,
                            batch_idx * len(x_train),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return (
            model.state_dict(),
            sum(epoch_loss) / len(epoch_loss),
            copy.deepcopy(model),
        )


class LocalUpdate_branch2(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.dataset = dataset[idxs]
        self.idxs = np.arange(0, len(self.dataset))
        self.local_bs = 50
        self.trainloader = DataLoader(
            MovieDataset(self.dataset, self.idxs),
            batch_size=self.local_bs,
            shuffle=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.learning_rate = 0.01
        self.criterion = nn.L1Loss().to(self.device)
        self.local_epochs = 1

    def update_weights(self, model, client_idx, global_round):

        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for iter in range(self.local_epochs):
            batch_loss = []
            for batch_idx, data_sample in enumerate(self.trainloader):

                x_train = data_sample[:, :-1].to(self.device)
                y_train = data_sample[:, -1].to(self.device)

                optimizer.zero_grad()
                outputs = model(x_train.unsqueeze(1))
                # outputs = model(X_tensor)
                loss = self.criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Round : {} | Vehicle : {} | Branch 2| Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            client_idx,
                            iter + 1,
                            batch_idx * len(x_train),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return (
            model.state_dict(),
            sum(epoch_loss) / len(epoch_loss),
            copy.deepcopy(model),
        )


def cache_hit_ratio(test_dataset, cache_items, request_num):
    request_items = test_dataset[:request_num, 1]
    count = Counter(request_items)
    CACHE_HIT_NUM = 0
    for item in cache_items:
        CACHE_HIT_NUM += count[item]
    CACHE_HIT_RATIO = CACHE_HIT_NUM / len(request_items) * 100

    return CACHE_HIT_RATIO
