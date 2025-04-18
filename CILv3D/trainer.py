import os
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *

class Trainer:
  def __init__(
      self,
      device: torch.device,
      model,
      model_path: str,
      train_loader: DataLoader,
      val_loader: Optional[DataLoader] = None,
      writer_path: Optional[str] = None,
      eval_epoch: bool = False,
      skip_training: bool = False,
      save_checkpoints: bool = False,
  ):
    self.device = device
    self.model = model
    self.model_path = model_path
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.writer_path = writer_path
    self.eval_epoch = eval_epoch
    self.skip_training = skip_training
    self.save_checkpoints = save_checkpoints

    if not writer_path:
      today = str(datetime.now()).replace(" ", "_")
      auto_name = "-".join([model_path.split('/')[-1].split('.')[0], today, f"lr_{LR}", f"bs_{BATCH_SIZE}"])
      writer_path = "runs/" + auto_name
    print("[+] Tensorboard output path:", writer_path)
    self.writer = SummaryWriter(writer_path)

  def save_checkpoint(self, min_loss):
    pass

  def train_step(self):
    pass

  def train(self):
    pass

  def eval_step(self):
    pass

  def eval(self):
    pass
