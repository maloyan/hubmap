import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import torch


from models import build_model

config = {
    "project": "HubMAP",
    "split_seed_list": [0],
    "FOLD_LIST": [0, 1, 2, 3],
    "model_path": "../input/hubmap-new-03-03/",
    "model_name": "unet_resnet34",
    "num_classes": 1,
    "resolution": (1024, 1024),  # (1024,1024),(512,512),
    "input_resolution": 512,  # (320,320), #(256,256), #(512,512), #(384,384)
    "deepsupervision": False,  # always false for inference
    "clfhead": False,
    "clf_threshold": 0.5,
    "small_mask_threshold": 0,  # 256*256*0.03, #512*512*0.03,
    "mask_threshold": 0.5,
    "pad_size": 256,  # (64,64), #(256,256), #(128,128)
    "tta": 3,
    "batch_size": 32,
    "FP16": False,
    "num_workers": 4,
    "device": "cuda",
    "input_path": "/kaggle/input/hubmap-organ-segmentation",
    "Adam": {
        "lr": 3e-5,
        #'betas':(0.9, 0.999),
        #'weight_decay':1e-5,
    },
    "lr_scheduler_name": "CosineAnnealingLR",
    "lr_scheduler": {
        "CosineAnnealingLR": {
            "T_max": 10,
            "eta_min": 1e-6
            # 'step_size_min':1e-6,
            # 't0':19,
            # 'tmult':1,
            # 'curr_epoch':-1,
            # 'last_epoch':-1,
        },
    },
    "reduce": 4,
    "num_epochs": 100,
    "class_labels": {
        0: "background",
        1: "cell"
    }
}

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

sample_submission = pd.read_csv(f"{config['input_path']}/sample_submission.csv")


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1)


transforms = A.Compose(
    [
        A.Normalize(),
        A.Lambda(image=to_tensor),
    ]
)

# model
model = build_model(
    resolution=config["resolution"],
    deepsupervision=config["deepsupervision"],
    clfhead=config["clfhead"],
    load_weights=False,
)

state_dict =torch.load('best_model.pt')
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v

model.load_state_dict(new_state_dict)


model.to('cuda')
model.eval()

def generate_new_shape(img, img_size):
    new_shape = (
        int(np.ceil(img.shape[1] / img_size) * img_size), 
        int(np.ceil(img.shape[2] / img_size) * img_size)
    )
    return new_shape

final_results = []
for idx in sample_submission.id.values:
    data = rasterio.open(
        f"{config['input_path']}/test_images/{idx}.tiff",
        transform=rasterio.Affine(1, 0, 0, 0, 1, 0),
        num_threads="all_cpus",
    )

    res = transforms(image=np.moveaxis(data.read([1, 2, 3]), 0, -1))
    image = torch.tensor(res['image'], dtype=torch.float)
    new_shape = generate_new_shape(image, config['input_resolution'])
    res_mask = np.zeros((new_shape[0], new_shape[1]))

    image_new = np.full((3, new_shape[0], new_shape[1]), image[0][0][0])
    image_new[:, :image.shape[1], :image.shape[2]] = image

    res_mask = np.zeros((image_new.shape[1], image_new.shape[2]))

    for i in range(0, image_new.shape[1], config['input_resolution']):
        for j in range(0, image_new.shape[2], config['input_resolution']):
            x_tensor = torch.tensor([image_new[:, i:i+config['input_resolution'], j:j+config['input_resolution']]], device='cuda')
            pr_mask = model(x_tensor)[0]
            pr_mask = torch.sigmoid(pr_mask).squeeze().detach().cpu().numpy()
            res_mask[i:i+config['input_resolution'], j:j+config['input_resolution']] = pr_mask

    res_mask = res_mask[:image.shape[1], :image.shape[2]]
    res_mask =  res_mask > 0.45
    pred_rle  = mask2rle(res_mask)
    final_results.append(pred_rle)
    sample_submission['rle'] = final_results
