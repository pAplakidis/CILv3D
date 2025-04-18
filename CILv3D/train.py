#!/usr/bin/env python3
import os
import psutil
import torch
from torch.utils.data import DataLoader

from dataset import *
from config import *

N_WORKERS = PREFETCH_FACTOR = psutil.cpu_count(logical=False)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available else "cpu")
  print("[+] Using device:", device)

  train_set = CarlaDataset(
    base_dir=DATA_DIR,
    townslist=TRAIN_TOWN_LIST,
    image_size=IMAGE_SIZE,
    use_imagenet_norm=USE_IMAGENET_NORM,
    sequence_size=SEQUENCE_SIZE
  )
  val_set = CarlaDataset(
    base_dir=DATA_DIR,
    townslist=EVAL_TOWN_LIST,
    image_size=IMAGE_SIZE,
    use_imagenet_norm=USE_IMAGENET_NORM,
    sequence_size=SEQUENCE_SIZE
  )

  train_loader =  DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                             prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=True)
  val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=True)
