import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_classes: int, 
                 conv1_channels=64, conv2_channels=128, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, conv1_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(conv1_channels)
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(conv2_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)
        self.flatten = nn.Flatten()
        # after two conv layers, shape is (batch, conv2_channels, seq_len)
        self.fc = nn.Linear(conv2_channels * seq_len, num_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))   # ➔ (batch, C1, L)
        x = self.relu(self.bn2(self.conv2(x)))   # ➔ (batch, C2, L)
        x = self.dropout(x)
        x = self.flatten(x)                      # ➔ (batch, C2*L)
        x = self.fc(x)                           # ➔ (batch, num_classes)
        return x

    