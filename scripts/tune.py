# scripts/tune.py

import os
import logging
import argparse
import torch
import optuna
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torch.optim import Adam, RMSprop, SGD
from pathlib import Path

# Make sure the project is installed in editable mode (pip install -e .)
# or adjust sys.path if necessary
try:
    from ml.data.dataset import EEGDataset
    from ml.models.eeg_mlp import EEGMLP
    from ml.models.cnn_1d import CNN1D
    from ml.engines.trainer import Trainer
except ImportError:
    import sys
    # Add the project root to the Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from ml.data.dataset import EEGDataset
    from ml.models.eeg_mlp import EEGMLP
    from ml.models.cnn_1d import CNN1D
    from ml.engines.trainer import Trainer

# --- Configuration ---
DEFAULT_DATA_ROOT = "./processed"
DEFAULT_EPOCHS = 20 # Number of epochs *per trial*
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_N_TRIALS = 50 # Number of HPO trials to run
DEFAULT_TUNING_LOG_DIR = "./logs/tuning"
DEFAULT_STUDY_NAME = "eeg_fatigue_tuning"

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Objective Function for Optuna ---
def objective(trial: optuna.trial.Trial, args: argparse.Namespace):
    """
    Defines a single training run with hyperparameters suggested by Optuna.
    """
    # --- Hyperparameters to Tune ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    # Model-specific hyperparameters
    if args.model_type == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dims = []
        for i in range(n_layers):
            out_features = trial.suggest_int(f"n_units_l{i}", 32, 256, log=True)
            hidden_dims.append(out_features)
        dropout_p = trial.suggest_float("dropout_p", 0.1, 0.7)
    elif args.model_type == "cnn":
        # Example: Tune number of channels in conv layers
        conv1_channels = trial.suggest_int("conv1_channels", 8, 64, log=True)
        conv2_channels = trial.suggest_int("conv2_channels", 16, 128, log=True)
        # Could also tune kernel_size, add more layers, tune pooling etc.
        # dropout_p = trial.suggest_float("dropout_p", 0.1, 0.7) # If CNN has dropout

    # --- Data Loading ---
    try:
        dataset = EEGDataset(args.data_root)
        n_total = len(dataset)
        if n_total == 0:
            raise ValueError(f"No data found in {args.data_root}. Check the path and data format.")

        n_val = int(n_total * args.val_split)
        n_train = n_total - n_val
        if n_train <= 0 or n_val <= 0:
             raise ValueError(f"Dataset split resulted in zero samples for train ({n_train}) or val ({n_val}). Check val_split ({args.val_split}) and total samples ({n_total}).")

        train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42)) # Use fixed seed for reproducibility *within* a trial split

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Infer input dimension from the first batch
        sample_x, _ = next(iter(train_loader))
        input_dim = sample_x.shape[-1] # Assuming shape is (batch, features) or (batch, channels, features) -> take last dim

    except FileNotFoundError:
        logging.error(f"Data root directory '{args.data_root}' not found.")
        raise
    except Exception as e:
        logging.error(f"Error during data loading/splitting: {e}")
        # Optuna needs a float return, indicate failure with a high value
        # Or re-raise to stop the study if data is fundamentally broken
        raise optuna.exceptions.TrialPruned(f"Data loading failed: {e}")


    # --- Model Instantiation ---
    try:
        if args.model_type == "mlp":
             if n_layers == 1:
                 if len(hidden_dims) != 2:
                      hidden_dims = (trial.suggest_int("n_units_l0", 32, 256, log=True),
                                     trial.suggest_int("n_units_l1", 32, 128, log=True))

                 model = EEGMLP(input_dim=input_dim, hidden_dims=tuple(hidden_dims), dropout_p=dropout_p)
                 print(f"Trial {trial.number}: MLP - lr={lr:.4e}, batch={batch_size}, opt={optimizer_name}, layers={hidden_dims}, dropout={dropout_p:.2f}")

        elif args.model_type == "cnn":
             model = CNN1D(input_dim=input_dim)
             # If CNN1D were updated to take conv channels:
             # model = CNN1D(input_dim=input_dim, conv1_channels=conv1_channels, conv2_channels=conv2_channels)
             print(f"Trial {trial.number}: CNN - lr={lr:.4e}, batch={batch_size}, opt={optimizer_name}") # Add tuned params if CNN is changed
        else:
             raise ValueError(f"Unknown model type: {args.model_type}")

        model = model.to(device)
    except Exception as e:
        logging.error(f"Error during model instantiation: {e}")
        raise optuna.exceptions.TrialPruned(f"Model instantiation failed: {e}")

    # --- Optimizer ---
    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # --- Criterion ---
    criterion = MSELoss()

    # --- Trainer ---
    trial_log_dir = Path(args.tuning_log_dir) / args.study_name / f"trial_{trial.number:04d}"
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # --- Training Loop ---
    try:
        best_val_loss = trainer.fit(n_epochs=args.epochs, trial=trial)

        if best_val_loss is None or torch.isnan(torch.tensor(best_val_loss)):
             logging.warning(f"Trial {trial.number} resulted in NaN/None loss. Pruning.")
             raise optuna.exceptions.TrialPruned("Resulted in NaN loss")

        return best_val_loss
    except optuna.exceptions.TrialPruned as e:
        raise e
    except Exception as e:
        logging.error(f"Error during training in trial {trial.number}: {e}", exc_info=True)
        return float('inf')


# --- Main Execution ---
if __name__ == "__main__":

    """
    Example to run mlp : python scripts/tune.py --model_type mlp --n_trials 50 
    Example to run cnn : python scripts/tune.py --model_type cnn --n_trials 30
    """

    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for EEG Fatigue Models")
    parser.add_argument("--model_type", type=str, required=True, choices=["mlp", "cnn"],
                        help="Type of model to tune ('mlp' or 'cnn')")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                        help="Path to the processed data directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of training epochs per trial")
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT,
                        help="Fraction of data to use for validation")
    parser.add_argument("--n_trials", type=int, default=DEFAULT_N_TRIALS,
                        help="Number of Optuna trials to run")
    parser.add_argument("--tuning_log_dir", type=str, default=DEFAULT_TUNING_LOG_DIR,
                        help="Base directory for saving tuning logs and checkpoints")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Name for the Optuna study. Defaults to '<model_type>_tuning'")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., 'sqlite:///tuning_results.db'). If None, uses in-memory storage.")

    args = parser.parse_args()

    if args.study_name is None:
        args.study_name = f"{args.model_type}_tuning"

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Optuna Study ---
    # Use a pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)

    # Create or load the study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",  # minimize mse
        storage=args.storage,
        pruner=pruner,
        load_if_exists=True
    )

    logging.info(f"Starting Optuna study '{args.study_name}' for model '{args.model_type}'")
    logging.info(f"Number of trials: {args.n_trials}")
    logging.info(f"Storage: {args.storage if args.storage else 'in-memory'}")
    logging.info(f"Device: {device}")

    # Run optimization
    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, timeout=None)
    except Exception as e:
        logging.error(f"An unexpected error occurred during tuning: {e}", exc_info=True)

    # --- Results ---
    logging.info(f"Tuning finished for study '{args.study_name}'.")
    logging.info(f"Number of finished trials: {len(study.trials)}")


    # --- Logging ---
    try:
        best_trial = study.best_trial
        logging.info("Best trial:")
        logging.info(f"  Value (Min Validation Loss): {best_trial.value:.6f}")
        logging.info("  Params: ")
        for key, value in best_trial.params.items():
            logging.info(f"    {key}: {value}")

        best_trial_log_dir = Path(args.tuning_log_dir) / args.study_name / f"trial_{best_trial.number:04d}"
        logging.info(f"  Best model checkpoint saved in: {best_trial_log_dir / 'best.pth'}")
    except ValueError:
        logging.warning("Could not find best trial. Did the study run successfully?")
    except Exception as e:
         logging.error(f"Error retrieving best trial info: {e}")

    # Example: Print all trials (optional)
    df = study.trials_dataframe()
    print("\n--- All Trials ---")
    print(df)