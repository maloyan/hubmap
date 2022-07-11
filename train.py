import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader

import wandb
from config import config
from dataset import HuBMAPDataset
from engine import run_train
from models import build_model

# wandb.init(
#     config=config,
#     project=config["project"],
#     name=f"{config['input_resolution']}_{config['model_name']}",
# )


accelerate = Accelerator(log_with="wandb")
accelerate.init_trackers(
    config["project"],
    config=config
)

train_df = pd.read_csv(f"{config['input_path']}/train.csv")
sub_df = pd.read_csv(f"{config['input_path']}/sample_submission.csv")


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1)


transforms = A.Compose(
    [   
        A.OneOf([
            A.CropNonEmptyMaskIfExists(
                width=config["input_resolution"], height=config["input_resolution"], p=0.6
            ),
            A.RandomCrop(
                width=config["input_resolution"], height=config["input_resolution"], p=0.4
            )
        ], p=1.0),
        A.Normalize(),
        A.Lambda(image=to_tensor),
    ]
)

train_dataset = HuBMAPDataset(train_df, transforms, data_len=10000)
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

# model = smp.UnetPlusPlus(
#     encoder_name='resnet34',
#     encoder_weights='imagenet',
#     in_channels=3,
#     classes=1,
#     activation=None, #'sigmoid'
# )

optimizer = optim.Adam(model.parameters(), **config["Adam"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, **config["lr_scheduler"]["CosineAnnealingLR"]
)

class SumTwoLosses(smp.utils.base.Loss):
    def __init__(self, l1, l2, a=1, b=1):
        super().__init__(name="SumTwoLosses")
        self.l1 = l1
        self.l2 = l2
        self.a = a
        self.b = b

    def __call__(self, *inputs):
        return self.a * self.l1.forward(*inputs) + self.b * self.l2.forward(*inputs)

seg_loss_func = SumTwoLosses(
    smp.losses.DiceLoss(mode='binary'), smp.losses.SoftBCEWithLogitsLoss()
)

model, optimizer, train_dataloader = accelerate.prepare(
    model, optimizer, train_dataloader
)

metrics = smp.utils.metrics.IoU(threshold=0.5)
# train_epoch = smp.utils.train.TrainEpoch(
#     model,
#     loss=seg_loss_func,
#     metrics=metrics,
#     optimizer=optimizer,
#     device='cuda',
#     verbose=True,
# )
# train_epoch.run(train_dataloader)
best_loss = 100
for epoch in range(config["num_epochs"]):
    train_info = run_train(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        seg_loss_func,
        metrics,
        accelerate,
    )
    if train_info["loss"] < best_loss:
        torch.save(model.state_dict(), "best_model.pt")
        best_loss = train_info["loss"]

    accelerate.log(
        {
            "loss": train_info["loss"],
            "iou": train_info["iou"],
            "images": wandb.Image(
                np.moveaxis(train_info["images"][0].detach().cpu().numpy(), 0, -1),
                masks={
                    "predictions": {
                        "mask_data": (torch.sigmoid(train_info["predictions"][0][0]).detach().cpu().numpy() > 0.5).astype(int),
                        "class_labels": config["class_labels"],
                    },
                    "ground_truth": {
                        "mask_data": train_info["masks"][0][0].detach().cpu().numpy(),
                        "class_labels": config["class_labels"],
                    },
                },
            ),
        }
    )

accelerate.end_training()
