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

    for images, masks in progress_bar:
        optimizer.zero_grad()
        prediction = model.forward(images)
        loss = seg_loss_func(prediction[0], masks)
        accelerate.backward(loss)

        #loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_description(
            f"loss: {loss.item():.4f} lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
    return loss.item()
