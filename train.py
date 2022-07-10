import albumentations as A
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader

from config import config
from dataset import HuBMAPDataset
from engine import run_train
from models import build_model

accelerate = Accelerator()

train_df = pd.read_csv(f"{config['input_path']}/train.csv")
sub_df = pd.read_csv(f"{config['input_path']}/sample_submission.csv")


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1)


transforms = A.Compose(
    [
        A.CropNonEmptyMaskIfExists(
            width=config["input_resolution"], height=config["input_resolution"]
        ),
        A.Normalize(),
        A.Lambda(image=to_tensor),
    ]
)

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


seg_loss_func = smp.utils.base.SumOfLosses(
    smp.utils.losses.DiceLoss(), smp.utils.losses.BCEWithLogitsLoss()
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
    train_loss = run_train(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        seg_loss_func,
        metrics,
        accelerate,
    )
    if train_loss < best_loss:
        torch.save(model.state_dict(), "best_model.pt")
