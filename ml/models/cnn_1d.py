import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel
        return self.model(x).squeeze(1)
        # x: (batch, 1, input_dim) ➔ (batch, 32, 1) ➔ (batch, 32) ➔ (batch, 1)
    