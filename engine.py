import gc
import pickle
import time
from os.path import join as opj

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from dataset import HuBMAPDataset
from losses import criterion_lovasz_hinge_non_empty
from lovasz_loss import lovasz_hinge
from metrics import dice_sum, dice_sum_2
from models import build_model
from scheduler import CosineLR

# output_path = config["OUTPUT_PATH"]
# fold_list = config["FOLD_LIST"]
# pretrain_path_list = config["pretrain_path_list"]
# device = config["device"]


def run_train(
    model, train_dataloader, optimizer, scheduler, seg_loss_func, accelerate
):  # , pseudo_df, trn_idxs_list, val_idxs_list):
    model.train()

    progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
    # tr_it = iter(train_dataloader)
    dataset_size = 0
    running_loss = 0.0
    for batch in progress_bar:
        inputs, masks, labels = (batch["img"], batch["mask"], batch["labels"])

        outputs = model(inputs)
        loss = seg_loss_func(masks, outputs[0][:, 0, :, :])
        accelerate.backward(loss)
        optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        running_loss += loss.item() * config["batch_size"]
        dataset_size += config["batch_size"]
        losses = running_loss / dataset_size
        progress_bar.set_description(
            f"loss: {losses:.4f} lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
        # del batch, inputs, masks, outputs, loss
    print(f"Train loss: {losses:.4f}")
