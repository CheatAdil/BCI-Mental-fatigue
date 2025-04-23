# ml/engines/trainer.py
import os
import numpy as np
import torch
import logging
import optuna
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from pathlib import Path

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader = None,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 work_dir: str = "./"):
        
        
        # each run logs to logs/<timestamp>/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs") / Path(f"{model.__class__.__name__}") / timestamp 
        log_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.setup_logging(log_dir)

        self.model        = model.to(device)
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = lr_scheduler
        self.device       = device
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.writer       = SummaryWriter(log_dir=os.path.join(work_dir, log_dir))
        self.best_metric  = float('inf')
        self.best_loss    = float('inf')

        logging.info(f"Trainer initialized. Model: {model.__class__.__name__}, Log Dir: {log_dir}")
        logging.info(f"with the following hyperparameters:")
        logging.info(f"  - batch size: {train_loader.batch_size}")
        logging.info(f"  - learning rate: {optimizer.param_groups[0]['lr']}")
        logging.info(f"  - optimizer: {optimizer.__class__.__name__}")
        logging.info(f"  - criterion: {criterion.__class__.__name__}")
        logging.info(f"  - device: {device}")
        logging.info(f"  - train dataset size: {len(train_loader.dataset)}")
        if val_loader:
             logging.info(f"  - val dataset size: {len(val_loader.dataset)}")


    def setup_logging(self, log_dir: Path):
        log_file = log_dir / "train.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x)
            loss  = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(self.train_loader)
        self.writer.add_scalar("train/loss", avg_loss, epoch)
        logging.info(f"[Train] Epoch {epoch} | Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch: int):
        if self.val_loader is None:
            return None, {} # Return loss and metrics dict
        self.model.eval()
        preds_all, targets_all = [], []
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [val]")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                loss  = self.criterion(preds, y)
                running_loss += loss.item()

                preds_all.append(preds.cpu().numpy())
                targets_all.append(y.cpu().numpy())

        # concat for metric calculation
        preds_all   = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)

        # compute regression metrics
        metrics = {}
        try:
            metrics['mse'] = mean_squared_error(targets_all, preds_all)
            metrics['mae'] = mean_absolute_error(targets_all, preds_all)
            metrics['r2']  = r2_score(targets_all, preds_all)
        except ValueError as e: # Handle potential errors if predictions are NaN/inf
             logging.error(f"Error calculating metrics: {e}. Setting metrics to NaN.")
             metrics['mse'] = float('nan')
             metrics['mae'] = float('nan')
             metrics['r2'] = float('nan')

        avg_loss = running_loss / len(self.val_loader)

        # log to TensorBoard
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        self.writer.add_scalar("val/mse", metrics['mse'], epoch)
        self.writer.add_scalar("val/mae", metrics['mae'], epoch)
        self.writer.add_scalar("val/r2", metrics['r2'], epoch)

        # Log final validation metrics for the epoch
        logging.info(f"[Val] Epoch {epoch} | Loss: {avg_loss:.4f} | MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")
        return avg_loss, metrics # Return loss and metrics dict


    def fit(self, n_epochs: int, checkpoint_freq: int = 1, trial: 'optuna.trial.Trial' = None): # Add trial argument
        best_val_loss_epoch = float('inf')
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch) # Unpack metrics

            if self.scheduler and val_loss is not None:
                self.scheduler.step(val_loss)

            # Checkpointing logic
            # Save checkpoint only if it's the best validation loss
            # metric = val_loss if val_loss is not None else train_loss
            if val_loss is not None and val_loss < self.best_loss:
                 self.best_loss = val_loss
                 self.save_checkpoint(epoch, best=True)
                 best_val_loss_epoch = val_loss # Store the best loss achieved
            # elif epoch % checkpoint_freq == 0:
            #     self.save_checkpoint(epoch, best=False)

            # Optuna Pruning: Report intermediate results
            if trial is not None and val_loss is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    logging.info(f"Trial pruned at epoch {epoch}.")
                    raise optuna.TrialPruned()

        # Return the best validation loss achieved during this fit call
        return best_val_loss_epoch


    def save_checkpoint(self, epoch: int, best: bool = False):
        state = {
            'epoch':      epoch,
            'model':      self.model.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'scheduler':  self.scheduler.state_dict() if self.scheduler else None,
            'best_loss':  self.best_loss,
        }
        cwd = str(self.writer.log_dir)
        fname = f"checkpoint-epoch{epoch}.pth"
        path = os.path.join(cwd, fname)
        torch.save(state, path)
        if best:
            best_path = os.path.join(cwd, "best.pth")
            torch.save(state, best_path)
            logging.info(f"Saved BEST checkpoint to {best_path} (Epoch {epoch}, Loss: {self.best_loss:.4f})")
        else:
            # print(f"Saved checkpoint to {path}\n")
            logging.info(f"Saved checkpoint to {path}\n")