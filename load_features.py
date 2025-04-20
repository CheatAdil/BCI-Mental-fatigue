import os
import random
import numpy as np
from pathlib import Path

class FeatureLoader:
    def __init__(self, base_dir="processed"):
        self.base_dir = Path(base_dir)
        self.current_features = None
        self.current_path = None
        
    def load_random_features(self):
        # Get all subdirectories
        subdirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            raise ValueError("No subdirectories found in the processed folder")
            
        # Randomly select a directory
        selected_dir = random.choice(subdirs)
        features_path = selected_dir / "features.npy"
        
        if not features_path.exists():
            raise FileNotFoundError(f"features.npy not found in {selected_dir}")
            
        # Load the features
        self.current_features = np.load(features_path)
        self.current_path = features_path
        return self.current_features
        
    def get_first_sample(self):
        if self.current_features is None:
            raise ValueError("No features loaded. Call load_random_features() first.")
        return self.current_features[0]

# Example usage
if __name__ == "__main__":
    loader = FeatureLoader()
    features = loader.load_random_features()
    print(f"Loaded features from: {loader.current_path}")
    print(f"Shape of features array: {features.shape}")
    print("\nFirst sample:")
    print(loader.get_first_sample()) 