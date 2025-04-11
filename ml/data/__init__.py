# ml/data/__init__.py

"""Data-loading utilities for EEG fatigue detection."""

from .dataset import EEGDataset
# from .datamodule import EEGDataModule

__all__ = ["EEGDataset", "EEGDataModule"]
