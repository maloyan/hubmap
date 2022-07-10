
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from config import config
from transforms import get_transforms_train, get_transforms_valid
from utils import rle2mask

# https://www.kaggle.com/datasets/thedevastator/hubmap-2022-256x256
mean = np.array([0.7720342, 0.74582646, 0.76392896])
std = np.array([0.24745085, 0.26182273, 0.25782376])

s_th = 40  # saturation blancking threshold
p_th = (
    1000 * (config["input_resolution"] // 256) ** 2
)  # threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape(shape, order="F")  # Needed to align to RLE direction


def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(" ".join(str(x) for x in runs))
    return encs


class HuBMAPDataset(Dataset):
    def __init__(self, df, transforms, sz=config["input_resolution"], reduce=config["reduce"]):
        self.df = df
        self.transforms = transforms
        self.reduce = reduce
        self.sz = reduce * sz

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = rasterio.open(
            f"{config['input_path']}/train_images/{self.df.iloc[idx]['id']}.tiff",
            transform=identity,
            num_threads="all_cpus",
        )
        encs = self.df.iloc[idx]["rle"]
        mask = rle_decode(encs, data.shape) if encs is not None else None

        res = self.transforms(image=np.moveaxis(data.read([1, 2, 3]), 0, -1), mask=mask)

        return torch.tensor(res["image"], dtype=torch.float), \
               torch.tensor([res["mask"]], dtype=torch.float),

        # return {
        #     "img": torch.tensor(res["image"], dtype=torch.float),
        #     "mask": torch.tensor([res["mask"]], dtype=torch.float),
        #     "labels": 0,
        # }
