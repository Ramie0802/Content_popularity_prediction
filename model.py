import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SemanticAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super(SemanticAggregator, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):  # x: [batch_size, seq_len, embedding_dim]

        for batch in x:
            for seq in batch:
                pass


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, dim_in, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(dim_in, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 9, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
