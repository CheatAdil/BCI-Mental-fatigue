import os
import random
import numpy as np
from collections import deque
from pathlib import Path

class EEGSimulator:
    def __init__(self, datasets_dir=None):
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        # Use the datasets folder in the same directory as the script
        self.datasets_dir = script_dir / "datasets" if datasets_dir is None else Path(datasets_dir)
        self.samples_queue = None
        self.current_file = None
        
        # Automatically load a random dataset
        self.load_random_dataset()
        
    def load_random_dataset(self):
        # Get all .npy files in the datasets directory
        npy_files = list(self.datasets_dir.glob("*.npy"))
        
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {self.datasets_dir}")
            
        # Randomly select a file
        selected_file = random.choice(npy_files)
        self.current_file = selected_file
        
        # Load the data
        data = np.load(selected_file)
        
        # Convert to list of samples and create a deque
        samples = [sample for sample in data]
        self.samples_queue = deque(samples)
        
        return len(samples)
        
    def get_next_sample(self):
        if self.samples_queue is None:
            raise ValueError("No dataset loaded. Call load_random_dataset() first.")
            
        if not self.samples_queue:
            raise ValueError("No samples available in the queue")
            
        # Get the first sample
        sample = self.samples_queue[0]
        
        # move this element to the end of the queue
        self.samples_queue.rotate(-1)
        
        return sample

# Example usage
if __name__ == "__main__":
    simulator = EEGSimulator()
    num_samples = simulator.load_random_dataset()
    print(f"Loaded {num_samples} samples from: {simulator.current_file}")
    
    # Get and print 5 samples to demonstrate the circular queue
    print("\nFirst 5 samples (will cycle through the dataset):")
    for i in range(5):
        sample = simulator.get_next_sample()
        print(f"Sample {i+1}:")
        print(sample)
        print("-" * 50)
