import os
import time
import torch
import numpy as np
from pathlib import Path

# Import your EEG simulator
from eeg_simulator.eeg_sim import EEGSimulator
# Import the model class you're using (either EEGMLP, CNN1D)
from ml.models.cnn_1d import CNN1D
from ml.models.eeg_mlp import EEGMLP


class MentalFatigueDetector:
    def __init__(self, model_path, input_dim=None, device=None, threshold=0.7):
        """
        Initialize the Mental Fatigue Detector

        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            input_dim: Input dimension for the model (optional, auto-detected if not provided)
            device: Device to run the model on (cpu or cuda)
            threshold: Threshold value for detecting fatigue (0.0-1.0)
        """
        self.threshold = threshold
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Initialize model and simulator
        self._setup_model()
        self._setup_simulator()

    def _setup_model(self):
        """Initialize the appropriate model based on the checkpoint path"""
        # Determine model type from path
        model_type = self.model_path.parts[1] if len(self.model_path.parts) > 1 else None

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Initialize model based on type
        if model_type == "CNN1D":
            self.model = CNN1D(input_dim=129)
            print(f"Initializing CNN1D model with input_dim=129")
        elif model_type == "EEGMLP":
            # Auto-detect input dimension if possible
            if 'net.0.weight' in checkpoint['model']:
                input_dim = checkpoint['model']['net.0.weight'].shape[1]
                print(f"Detected input dimension from EEGMLP model: {input_dim}")
            else:
                input_dim = 121
            self.model = EEGMLP(input_dim=input_dim, hidden_dims=(128, 64))
        else:
            # Fallback: try to infer model type from checkpoint structure
            if 'net.0.weight' in checkpoint['model']:
                self.model = EEGMLP(input_dim=121, hidden_dims=(128, 64))
                print("Model type not determined from path, defaulting to EEGMLP")
            else:
                self.model = CNN1D(input_dim=129)
                print("Model type not determined from path, defaulting to CNN1D")

        # Load weights and prepare model
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {self.model_path} (epoch {checkpoint['epoch']})")

    def _setup_simulator(self):
        """Initialize the EEG simulator"""
        self.simulator = EEGSimulator()
        sample_count = self.simulator.load_random_dataset()
        print(f"Loaded simulator with {sample_count} samples")
        print(f"Using data from: {self.simulator.current_file}")

    def preprocess_sample(self, sample):
        """
        Preprocess the raw EEG sample to match the expected input format of the model
        """
        # Convert to tensor
        sample_tensor = torch.tensor(sample, dtype=torch.float32).to(self.device)
        if len(sample_tensor.shape) == 1:
            sample_tensor = sample_tensor.unsqueeze(0)  # Add batch dimension

        return sample_tensor

    def predict(self, sample):
        """Get prediction from the model"""
        with torch.no_grad():
            prediction = self.model(sample)
        return prediction.item()

    def is_fatigued(self, prediction):
        """Determine if the prediction indicates fatigue"""
        return prediction >= self.threshold

    def run(self, interval=1.0, max_iterations=None):
        """
        Main loop to continuously monitor for mental fatigue

        Args:
            interval: Time interval between checks (in seconds)
            max_iterations: Maximum number of iterations (None for indefinite)
        """
        iterations = 0

        try:
            print("Starting mental fatigue monitoring...")
            print(f"Fatigue threshold set to: {self.threshold}")
            print("Press Ctrl+C to stop monitoring")
            print("-" * 50)

            while max_iterations is None or iterations < max_iterations:
                # Get next EEG sample
                eeg_sample = self.simulator.get_next_sample()

                # Preprocess the sample
                processed_sample = self.preprocess_sample(eeg_sample)

                # Get prediction
                fatigue_score = self.predict(processed_sample)

                # Check if fatigue is detected
                if self.is_fatigued(fatigue_score):
                    print("\n🚨 ALERT: MENTAL FATIGUE DETECTED! 🚨")
                    print(f"Fatigue score: {fatigue_score:.4f}")
                    print("Consider taking a break.\n")
                else:
                    print(f"Current fatigue score: {fatigue_score:.4f}")

                iterations += 1
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")

        print(f"Monitored {iterations} samples.")


def find_latest_model():
    # Find the most recent trained model in the logs directory
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("No logs directory found. Have you trained a model yet?")

    # Find all model directories
    model_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        raise FileNotFoundError("No model directories found in logs/")

    # For each model directory, find the most recent run
    latest_model_path = None
    latest_timestamp = 0

    for model_dir in model_dirs:
        run_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        for run_dir in run_dirs:
            # Check if the run directory name is a timestamp
            try:
                dir_name = run_dir.name
                timestamp = int(dir_name.replace("_", ""))
                if timestamp > latest_timestamp:
                    # Check if this run has a best.pth file
                    best_model_path = run_dir / "best.pth"
                    if best_model_path.exists():
                        latest_model_path = best_model_path
                        latest_timestamp = timestamp
            except ValueError:
                # Skip directories that don't have timestamp-like names
                continue

    if latest_model_path is None:
        raise FileNotFoundError("No best.pth found in any logs subdirectory")

    return latest_model_path


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mental Fatigue Detection from EEG")
    parser.add_argument("--model", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--input_dim", type=int, default=129, help="Input dimension for the model")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for fatigue detection")
    parser.add_argument("--interval", type=float, default=1.0, help="Time interval between samples (seconds)")
    args = parser.parse_args()

    # Determine model path - from command line or find latest
    if args.model:
        model_path = args.model
        print(f"Using model specified at: {model_path}")
    else:
        try:
            model_path = find_latest_model()
            print(f"Found latest model at: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify the path to a trained model checkpoint using --model.")
            exit(1)

    # Create the detector
    detector = MentalFatigueDetector(
        model_path=model_path,
        input_dim=args.input_dim,
        threshold=args.threshold
    )

    # Start monitoring
    detector.run(interval=args.interval)