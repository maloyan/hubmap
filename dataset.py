import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from config import config


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


class HuBMAPDataset(Dataset):
    def __init__(self, df, transforms, data_len=1000):
        self.df = df
        self.transforms = transforms
        self.data_len = data_len
        self.data_shape = df.shape[0]
    def __len__(self):
        return self.data_len #self.df.shape[0]

    def __getitem__(self, idx):
        idx = idx % self.data_shape
        data = rasterio.open(
            f"{config['input_path']}/train_images/{self.df.iloc[idx]['id']}.tiff",
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),
            num_threads="all_cpus",
        )
        encs = self.df.iloc[idx]["rle"]
        mask = rle_decode(encs, data.shape) if encs is not None else None

        res = self.transforms(image=np.moveaxis(data.read([1, 2, 3]), 0, -1), mask=mask)

        return (
            torch.tensor(res["image"], dtype=torch.float),
            torch.tensor(res["mask"], dtype=torch.float).unsqueeze(0),
        )

        # return {
        #     "img": torch.tensor(res["image"], dtype=torch.float),
        #     "mask": torch.tensor([res["mask"]], dtype=torch.float),
        #     "labels": 0,
        # }
