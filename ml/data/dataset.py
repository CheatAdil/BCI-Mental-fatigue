import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """
    assumes root_dir has subfolders, each containing:
      - features.npy  : shape (N_samples, n_channels, seq_len)
      - labels.npy    : shape (N_samples,)
    """
    def __init__(self, root_dir):
        self.samples = []
        for subj in os.listdir(root_dir):
            folder = os.path.join(root_dir, subj)
            if not os.path.isdir(folder):
                continue
            features = np.load(os.path.join(folder, 'features.npy'))  # e.g. (n_s, C, L)
            labels  = np.load(os.path.join(folder, 'labels.npy'))    # e.g. (n_s,)
            assert features.shape[0] == labels.shape[0]
            for i in range(features.shape[0]):
                self.samples.append((features[i], labels[i]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
