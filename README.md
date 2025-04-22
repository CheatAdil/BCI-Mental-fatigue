# Brain Computer Interface (CSCI 490) Course Project -- Mental Fatigue Project 

# Running the EEG Regression Training Script

This document explains how to install and run the EEG feature–based regression training pipeline, including:

- Installing the project in editable mode (`pip install -e .`)
- Preparing your data directory
- Launching training via `python scripts/train.py`
- Monitoring with TensorBoard
- Customizing and future use

---

## 📦 1. Installation

1. **Clone the repository** (if you haven’t already):
   ```bash
   git clone <your-repo-url>
   cd my_eeg_project
   ```

2. **Install in editable (“development”) mode**:
   ```bash
   pip install -e .
   ```
   - The `-e` (editable) flag tells `pip` to link the source directory into your Python environment, so any changes you make to `ml/` are immediately available without reinstallation.
   - It also registers `ml` as a top‑level package, allowing smooth imports (`import ml.data`, `import ml.models`, etc.).

3. **Verify the install**:
   ```bash
   python -c "import ml; print('ml package loaded:', ml)"
   ```

---

## 📂 2. Data Preparation

Place your processed EEG feature data under the `processed/` folder at the project root. The expected layout is:

```
processed/
├── subject1/
│   ├── features.npy    # shape (n_samples, n_features)
│   └── labels.npy      # shape (n_samples,) continuous targets
├── subject2/
│   ├── features.npy
│   └── labels.npy
└── ...
```

Each subject subfolder must contain matching `features.npy` and `labels.npy` files.

---

## ▶️ 3. Launch Training

From the project root, simply run:

```bash
python scripts/train.py
```

What happens:

1. A new log directory is created at `logs/<YYYYMMDD_HHMMSS>/` for this run.
2. The training loop runs for the default number of epochs (20), computing **MSE** loss and logging **MSE**, **MAE**, and **R²** on the validation set each epoch.
3. Checkpoints are saved in the log folder (`checkpoint-epochX.pth` and `best.pth`).
4. Progress bars show per‑batch train/validation loss.

---

<!-- ## 📊 4. Monitoring with TensorBoard

To visualize training curves, open another terminal and run:

```bash
tensorboard --logdir logs
```

Then open the displayed URL (e.g. http://localhost:6006) in your browser to see:

- **Train loss** (MSE) over epochs
- **Val loss**, **Val MSE**, **MAE**, **R²** metrics over epochs

--- -->

## ⚙️ 5. Customization & Future Runs

Hyperparameters are defined at the top of `scripts/train.py`:

```python
# scripts/train.py

data_root   = "./processed"
batch_size  = 64
lr          = 1e-3
num_epochs  = 20
val_split   = 0.2
hidden_dims = (128, 64)
```

Feel free to adjust these values or extend the script to accept **command‑line arguments** (e.g. via `argparse`).

Once installed in editable mode, you can:

- Rerun training after edits without reinstalling.
- Import modules in new scripts or interactive sessions:
  ```python
  from ml.data.dataset import EEGDataset
  from ml.models.eeg_mlp import EEGMLP
  ```

---

## 🧹 6. Uninstalling

If you ever want to clean up:

```bash
pip uninstall my-eeg-project
```

This removes the editable link and package registration.

---

Happy modeling! 🎉




---
## 🧠7. Mental Fatigue Detection
After training your model, you can use the real-time mental fatigue detection script to continuously monitor EEG signals for signs of fatigue.

From the project root, run:

```bash
python main.py
```

The script will automatically find and load the most recently trained model

### Command-line Options

You can customize the detection script with several options:

```bash
python main.py --model /path/to/model.pth --threshold 0.7 --interval 1.0
````
| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to a specific model checkpoint | Auto-detects latest model |
| `--input_dim` | Input dimension for the model | 129 |
| `--threshold` | Fatigue detection threshold (0.0-1.0) | 0.7 |
| `--interval` | Time between samples (seconds) | 1.0 |

## Full hierarchy

```
BCI-MENTAL-FATIGUE/
├── processed/                     # raw & processed data (not under version control)
│   └── <subject_name>/
│       ├── eeg.npy
│       ├── eog.npy  
│       ├── features.npy
│       └── labels.npy 
│
├── ml/                       # Python package for your code
│   ├── data/                 # data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py        # EEGDataset class
│   │   └── datamodule.py     # optional LightningDataModule or similar
│   │
│   ├── models/               # model definitions
│   │   ├── __init__.py
│   │   └── eeg_net.py        # EEGNet class
│   │
│   ├── engines/              # training / evaluation logic
│   │   ├── __init__.py
│   │   └── trainer.py        # Trainer class or functions
│   │
│   ├── utils/                # helper functions (metrics, plotting)
│   │   ├── __init__.py
│   │   └── utils.py
│   │
│   └── config/               # configuration files (YAML, JSON)
│       └── default.yaml
│
├── scripts/                  # entry‐point scripts
│   ├── train.py              # calls ml.engines.trainer
│   └── evaluate.py           # runs inference & computes metrics
│
├── tests/                    # unit & integration tests
│   └── test_dataset.py
│
├── requirements.txt
├── setup.py   or   pyproject.toml
└── README.md
```