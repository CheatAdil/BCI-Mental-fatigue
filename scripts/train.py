# scripts/train.py

import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torch.optim import Adam

from ml.data.dataset import EEGDataset
from ml.models.eeg_mlp import EEGMLP
from ml.models.cnn_1d import CNN1D
from ml.engines.trainer import Trainer

# --- Config ---
data_root   = "./processed"
batch_size  = 64
lr          = 1e-3
num_epochs  = 20
val_split   = 0.2
hidden_dims = (128, 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
dataset = EEGDataset(data_root)
n_total = len(dataset)
n_val   = int(n_total * val_split)
n_train = n_total - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

# --- Model ---
# infer input dimension
sample_x, _ = next(iter(train_loader))
input_dim   = sample_x.shape[1]

# model     = EEGMLP(input_dim=input_dim, hidden_dims=hidden_dims)
model     = CNN1D(input_dim=input_dim)
model     = model.to(device)

# --- Training ---
criterion = MSELoss()           # regression loss
optimizer = Adam(model.parameters(), lr=lr)

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader
)

trainer.fit(n_epochs=num_epochs)
