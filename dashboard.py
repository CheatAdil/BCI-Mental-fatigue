import os
import time
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import threading
from pathlib import Path
import matplotlib
from eeg_simulator.eeg_sim import EEGSimulator
from ml.models.cnn_1d import CNN1D
from ml.models.eeg_mlp import EEGMLP

matplotlib.use("TkAgg")  # Use TkAgg backend


class MentalFatigueDashboard:
    def __init__(self, root, model_path, input_dim=None, threshold=0.7, history_size=100):
        """
        Initialize the Mental Fatigue Dashboard

        Args:
            root: Tkinter root window
            model_path: Path to the trained model checkpoint
            input_dim: Input dimension for the model (optional)
            threshold: Threshold value for detecting fatigue (0.0-1.0)
            history_size: Number of history points to store
        """
        self.root = root
        self.root.title("Mental Fatigue Monitor")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        self.threshold = threshold
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data storage
        self.history_size = history_size
        self.fatigue_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.eeg_data = None
        self.latest_score = 0
        self.is_monitoring = False
        self.stop_event = threading.Event()

        # Load model and simulator
        self._setup_model()
        self._setup_simulator()

        # Setup UI
        self._setup_ui()

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

    def _setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(header_frame, text="Mental Fatigue Monitor", font=("Arial", 18, "bold"))
        title_label.pack(side=tk.LEFT)

        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Threshold control
        threshold_frame = ttk.Frame(controls_frame)
        threshold_frame.pack(fill=tk.X, pady=5)

        threshold_label = ttk.Label(threshold_frame, text="Fatigue Threshold:")
        threshold_label.pack(side=tk.LEFT, padx=(0, 5))

        self.threshold_var = tk.DoubleVar(value=self.threshold)

        # Create the threshold scale without resolution parameter
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0,
                                    variable=self.threshold_var, orient=tk.HORIZONTAL,
                                    length=200)
        threshold_scale.pack(side=tk.LEFT)

        self.threshold_display = tk.StringVar(value=f"{self.threshold:.1f}")
        threshold_value = ttk.Label(threshold_frame, textvariable=self.threshold_display)
        threshold_value.pack(side=tk.LEFT, padx=5)

        # Button frame
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.start_button = ttk.Button(button_frame, text="Start Monitoring",
                                       command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = ttk.Button(button_frame, text="Stop Monitoring",
                                      command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        # Current score frame
        score_frame = ttk.LabelFrame(main_frame, text="Current Fatigue Level", padding=10)
        score_frame.pack(fill=tk.X, pady=(0, 10))

        # Gauge for current score
        self.gauge_fig = Figure(figsize=(4, 2), dpi=100)
        self.gauge_ax = self.gauge_fig.add_subplot(111)
        self.gauge_canvas = FigureCanvasTkAgg(self.gauge_fig, master=score_frame)
        self.gauge_canvas.get_tk_widget().pack(fill=tk.BOTH)
        self._update_gauge(0)

        # Current status and alert
        self.status_var = tk.StringVar(value="Not monitoring")
        self.status_label = ttk.Label(score_frame, textvariable=self.status_var,
                                      font=("Arial", 14))
        self.status_label.pack(pady=5)

        # History graph frame
        history_frame = ttk.LabelFrame(main_frame, text="Fatigue History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # History graph
        self.history_fig = Figure(figsize=(5, 3), dpi=100)
        self.history_ax = self.history_fig.add_subplot(111)
        self.history_ax.set_ylim(0, 1)
        self.history_ax.set_xlabel("Time (s)")
        self.history_ax.set_ylabel("Fatigue Score")
        self.history_ax.grid(True)

        # Add threshold line to history graph
        self.threshold_line = self.history_ax.axhline(y=self.threshold, color='r', linestyle='--')

        self.history_canvas = FigureCanvasTkAgg(self.history_fig, master=history_frame)
        self.history_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # EEG Signal frame
        eeg_frame = ttk.LabelFrame(main_frame, text="EEG Signal", padding=10)
        eeg_frame.pack(fill=tk.BOTH, expand=True)

        # EEG graph
        self.eeg_fig = Figure(figsize=(5, 2), dpi=100)
        self.eeg_ax = self.eeg_fig.add_subplot(111)
        self.eeg_ax.set_xlabel("Channel")
        self.eeg_ax.set_ylabel("Amplitude")
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, master=eeg_frame)
        self.eeg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.statusbar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Update threshold when slider changes
        self.threshold_var.trace_add("write", self._on_threshold_changed)

        # Apply styling to the window
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0")
        style.configure("TLabelframe", background="#f0f0f0")
        style.configure("TLabelframe.Label", background="#f0f0f0")

    def _on_threshold_changed(self, *args):
        """Update threshold value when slider changes"""
        # Get the raw value from slider
        raw_value = self.threshold_var.get()

        # Round to nearest 0.1
        rounded_value = round(raw_value * 10) / 10

        # Update the threshold value
        self.threshold = rounded_value

        # Update the display with one decimal place
        self.threshold_display.set(f"{rounded_value:.1f}")

        # Remove old threshold line and create a new one
        self.threshold_line.remove()
        self.threshold_line = self.history_ax.axhline(y=self.threshold, color='r', linestyle='--')
        self.history_canvas.draw_idle()


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

    def _update_gauge(self, value):
        """Update the gauge visualization with the current fatigue value"""
        self.gauge_ax.clear()

        # Create a semi-circle gauge
        self.gauge_ax.set_xlim(-1.1, 1.1)
        self.gauge_ax.set_ylim(-0.1, 1.1)

        # Background
        bg = plt.Circle((0, 0), 1, fill=True, color='lightgray')
        self.gauge_ax.add_artist(bg)

        # Determine color based on value
        if value < self.threshold * 0.5:
            color = 'green'
        elif value < self.threshold:
            color = 'yellow'
        else:
            color = 'red'

        # Foreground (value indicator)
        theta = np.linspace(-np.pi, -np.pi + value * np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        self.gauge_ax.fill_between(x, 0, y, color=color, alpha=0.8)

        # Add value text
        self.gauge_ax.text(0, 0.5, f"{value:.2f}", ha='center', va='center',
                           fontsize=16, fontweight='bold')

        self.gauge_ax.set_aspect('equal')
        self.gauge_ax.axis('off')

        self.gauge_canvas.draw_idle()

    def _update_history_graph(self):
        """Update the history graph with the latest data"""
        self.history_ax.clear()

        # Set up the graph
        self.history_ax.set_ylim(0, 1)
        self.history_ax.set_xlabel("Time (s)")
        self.history_ax.set_ylabel("Fatigue Score")
        self.history_ax.grid(True)

        # Add threshold line
        self.threshold_line = self.history_ax.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')

        # Plot the history
        if len(self.fatigue_history) > 1:
            # Convert to relative time (seconds from start)
            relative_times = [t - self.time_history[0] for t in self.time_history]

            # Determine colors based on fatigue level
            colors = ['red' if score >= self.threshold else 'green' for score in self.fatigue_history]

            # Create scatter plot with colored points
            self.history_ax.scatter(relative_times, self.fatigue_history, c=colors, s=20)

            # Connect points with a line
            self.history_ax.plot(relative_times, self.fatigue_history, color='blue', alpha=0.5)

            self.history_ax.legend(['Threshold', 'Fatigue Level'])

        self.history_canvas.draw_idle()

    def _update_eeg_graph(self, eeg_data):
        """Update the EEG signal graph"""
        self.eeg_ax.clear()

        # Set up the graph
        self.eeg_ax.set_xlabel("Channel")
        self.eeg_ax.set_ylabel("Amplitude")

        # Plot the EEG data
        if eeg_data is not None:
            x = np.arange(len(eeg_data))
            self.eeg_ax.plot(x, eeg_data, color='blue')

            # Add a grid
            self.eeg_ax.grid(True, linestyle='--', alpha=0.7)

            # Set y-limits with some padding
            eeg_min = np.min(eeg_data)
            eeg_max = np.max(eeg_data)
            padding = (eeg_max - eeg_min) * 0.1
            self.eeg_ax.set_ylim(eeg_min - padding, eeg_max + padding)

        self.eeg_canvas.draw_idle()

    def _monitoring_loop(self):
        """Background thread for continuous monitoring"""
        start_time = time.time()

        try:
            while not self.stop_event.is_set():
                # Get next EEG sample
                eeg_sample = self.simulator.get_next_sample()
                self.eeg_data = eeg_sample  # Store for display

                # Preprocess the sample
                processed_sample = self.preprocess_sample(eeg_sample)

                # Get prediction
                fatigue_score = self.predict(processed_sample)
                self.latest_score = fatigue_score

                # Update history
                current_time = time.time()
                self.fatigue_history.append(fatigue_score)
                self.time_history.append(current_time)

                # Update UI (from main thread)
                is_fatigued = self.is_fatigued(fatigue_score)
                self.root.after(0, self._update_ui, fatigue_score, is_fatigued)

                # Sleep briefly to avoid using too much CPU
                time.sleep(1.0)

        except Exception as error:
            print(f"Error in monitoring loop: {error}")
            self.root.after(0, self._on_error, str(error))

    def _update_ui(self, fatigue_score, is_fatigued):
        """Update UI components with latest data (called from main thread)"""
        # Update gauge
        self._update_gauge(fatigue_score)

        # Update status
        if is_fatigued:
            self.status_var.set("⚠️ FATIGUE DETECTED! ⚠️")
            self.status_label.configure(foreground="red")
            # Flash the window to alert the user
            self.root.attributes('-topmost', 1)
            self.root.attributes('-topmost', 0)
        else:
            self.status_var.set("Normal Alertness")
            self.status_label.configure(foreground="green")

        # Update history graph
        self._update_history_graph()

        # Update EEG graph
        self._update_eeg_graph(self.eeg_data)

        # Update status bar
        elapsed = time.time() - self.time_history[0] if self.time_history else 0
        self.statusbar.config(
            text=f"Monitoring: {len(self.fatigue_history)} samples collected | Elapsed: {elapsed:.1f}s")

    def _on_error(self, error_msg):
        """Handle errors from the monitoring thread"""
        self.stop_monitoring()
        messagebox.showerror("Error", f"Monitoring stopped due to an error:\n{error_msg}")

    def start_monitoring(self):
        """Start the monitoring process"""
        if not self.is_monitoring:
            # Clear history
            self.fatigue_history.clear()
            self.time_history.clear()

            # Reset stop event
            self.stop_event.clear()

            # Start monitoring thread
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()

            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Monitoring...")
            self.statusbar.config(text="Monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring process"""
        if self.is_monitoring:
            self.stop_event.set()
            if hasattr(self, 'monitoring_thread'):
                self.monitoring_thread.join(timeout=1.0)
            self.is_monitoring = False

            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Monitoring stopped")
            self.statusbar.config(text="Monitoring stopped")


def find_latest_model():
    """Find the most recent trained model in the logs directory"""
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
                    # Check if this run has best.pth file
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
    from tkinter import messagebox

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mental Fatigue Detection Dashboard")
    parser.add_argument("--model", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--input_dim", type=int, default=129, help="Input dimension for the model")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for fatigue detection")
    args = parser.parse_args()

    # Create Tkinter root
    root = tk.Tk()


    # Handle window close


    def on_closing():
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        # Determine model path - from command line or find latest
        if args.model:
            model_path = args.model
            print(f"Using model specified at: {model_path}")
        else:
            try:
                model_path = find_latest_model()
                print(f"Found latest model at: {model_path}")
            except FileNotFoundError as e:
                messagebox.showerror("Error", f"{e}\nPlease specify a model path using --model.")
                root.destroy()
                exit(1)

        # Create the dashboard
        dashboard = MentalFatigueDashboard(
            root=root,
            model_path=model_path,
            input_dim=args.input_dim,
            threshold=args.threshold
        )

        # Start the Tkinter event loop
        root.mainloop()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        root.destroy()
