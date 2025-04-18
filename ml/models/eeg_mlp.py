# ml/models/eeg_mlp.py

import torch.nn as nn

class EEGMLP(nn.Module):
    """
    Simple MLP for EEG regression.
    """
    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dims[1], 1)  # single continuous output
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   # → (batch,)
