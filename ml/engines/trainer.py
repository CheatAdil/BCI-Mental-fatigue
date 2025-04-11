import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
        
        self.model        = model.to(device)
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = lr_scheduler
        self.device       = device
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.writer       = SummaryWriter(log_dir=os.path.join(work_dir, "logs"))
        self.best_metric  = float('inf')


    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]")
        for batch_idx, (x, y) in enumerate(pbar, 1):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss   = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                avg = running_loss / batch_idx
                pbar.set_postfix(loss=avg)
                self.writer.add_scalar("train/loss", avg, epoch * len(self.train_loader) + batch_idx)
        return running_loss / len(self.train_loader)

    def validate(self, epoch: int): 
        self.model.eval()
        running_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [val]")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss   = self.criterion(logits, y)
                running_loss += loss.item()
                if batch_idx % 10 == 0:
                    avg = running_loss / batch_idx
                    pbar.set_postfix(loss=avg)
                    self.writer.add_scalar("val/loss", avg, epoch * len(self.val_loader) + batch_idx)
        return running_loss / len(self.val_loader)
    
    def fit(self, n_epochs: int, checkpoint_freq: int = 1):
        for epoch in range(1, n_epochs+1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate(epoch)
            if self.scheduler:
                self.scheduler.step(val_loss if val_loss is not None else train_loss)
            # checkpointing
            metric = val_loss if val_loss is not None else train_loss
            if metric < self.best_metric:
                self.best_metric = metric
                self.save_checkpoint(epoch, is_best=True)
            elif epoch % checkpoint_freq == 0:
                self.save_checkpoint(epoch, is_best=False)

    def save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            'epoch':      epoch,
            'model':      self.model.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'scheduler':  self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
        }
        filename = f"checkpoint-epoch{epoch}.pth"
        path = os.path.join(self.writer.log_dir, filename)
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.writer.log_dir, "best.pth")
            torch.save(state, best_path)
            print(f"Saved best model to {best_path}")
        else:   
            print(f"Saved checkpoint to {path}")
        self.writer.add_scalar("train/loss", state['best_metric'], epoch)   