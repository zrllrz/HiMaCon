import os
import torch
# from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from pytorch_lightning import Callback


class MySaveLogger(Callback):
    def __init__(self, path, iter_freq=200):
        super().__init__()
        self.path = path
        self.iter_freq = iter_freq
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if trainer.is_global_zero and ((trainer.global_step + 1) % self.iter_freq == 0):
            ckpt_path = f"{trainer.global_step + 1}.pth"
            torch.save({'module': pl_module.state_dict()},
                       f"{self.path}/{ckpt_path}")
            print(f"Save ckpt: {self.path}/{ckpt_path}")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            ckpt_path = f"final-{trainer.global_step + 1}.pth"
            torch.save({'module': pl_module.state_dict()},
                       f"{self.path}/{ckpt_path}")
            print(f"Save ckpt: {self.path}/{ckpt_path}")
