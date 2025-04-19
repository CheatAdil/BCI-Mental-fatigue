# ./ml/models/__init__.py

"""Model utilities for EEG fatigue detection."""

from .eeg_net import EEGNet
from .cnn_1d import CNN1D
from .eeg_mlp import EEGMLP

__all__ = ["EEGNet", "CNN1D", "EEGMLP"]
