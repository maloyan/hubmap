import gc
import math
import os
import random
import sys
import time

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretrainedmodels
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from config import config
from dataset import HuBMAPDataset
from engine import run_train
from models import build_model
from scheduler import CosineLR
from utils import mask2rle

from accelerate import Accelerator


accelerate = Accelerator()

train_df = pd.read_csv(f"{config['input_path']}/train.csv")
sub_df = pd.read_csv(f"{config['input_path']}/sample_submission.csv")

# criterion = nn.BCEWithLogitsLoss().to(config["device"])
# criterion_clf = nn.BCEWithLogitsLoss().to(config["device"])

# add pseudo label
# if pseudo_df is not None:
#     trn_df = pd.concat([trn_df, pseudo_df], axis=0).reset_index(drop=True)

# dataloader
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1)

transforms = A.Compose([
        A.CropNonEmptyMaskIfExists(width=config["input_resolution"], height=config["input_resolution"]),
        A.Normalize(),
        A.Lambda(image=to_tensor),
])

train_dataset = HuBMAPDataset(train_df, transforms)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

# model
model = build_model(
    resolution=config["resolution"],
    deepsupervision=config["deepsupervision"],
    clfhead=config["clfhead"],
    load_weights=True,
)

# if pretrain_path_list is not None:
#     model.load_state_dict(torch.load(pretrain_path_list[fold]))

#         for p in model.parameters():
#             p.requires_grad = True

optimizer = optim.Adam(model.parameters(), **config["Adam"])
# optimizer = optim.RMSprop(model.parameters(), **config['RMSprop'])


scheduler = CosineLR(optimizer, **config["lr_scheduler"]["CosineAnnealingLR"])


class SumTwoLosses(smp.utils.base.Loss):
    def __init__(self, l1, l2, a=1, b=1):
        super().__init__(name="SumTwoLosses")
        self.l1 = l1
        self.l2 = l2
        self.a = a
        self.b = b

    def __call__(self, *inputs):
        return self.a * self.l1.forward(*inputs) + self.b * self.l2.forward(*inputs)


seg_loss_func = SumTwoLosses(smp.utils.losses.DiceLoss(), smp.utils.losses.BCELoss())

model, optimizer, train_dataloader = accelerate.prepare(
    model, optimizer, train_dataloader
)

val_score_best = -1e99
val_score_best2 = -1e99
loss_val_best = 1e99
epoch_best = 0
counter_ES = 0
trn_score = 0
trn_score_each = 0

for epoch in range(config["num_epochs"]):
    run_train(
        model, 
        train_dataloader, 
        optimizer, 
        scheduler, 
        seg_loss_func, 
        accelerate
    )
