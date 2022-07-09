config = {
    'split_seed_list':[0],
    'FOLD_LIST':[0,1,2,3],
    'model_path':'../input/hubmap-new-03-03/',
    'model_name':'seresnext101',
    'num_classes':1,
    'resolution':(1024, 1024), #(1024,1024),(512,512),
    'input_resolution':320, #(320,320), #(256,256), #(512,512), #(384,384)
    'deepsupervision':False, # always false for inference
    'clfhead':False,
    'clf_threshold':0.5,
    'small_mask_threshold':0, #256*256*0.03, #512*512*0.03,
    'mask_threshold':0.5,
    'pad_size':256, #(64,64), #(256,256), #(128,128)
    'tta':3,
    'batch_size':12,
    'FP16':False,
    'num_workers':4,
    'device': 'cuda',
    'input_path': '/kaggle/input/hubmap-organ-segmentation',
    'Adam':{
        'lr':1e-4, 
        'betas':(0.9, 0.999),
        'weight_decay':1e-5,
    },
    'lr_scheduler_name':'CosineAnnealingLR',
    'lr_scheduler':{
        'CosineAnnealingLR':{
            'step_size_min':1e-6,
            't0':19,
            'tmult':1,
            'curr_epoch':-1,
            'last_epoch':-1,
        },
    },
    "reduce": 4,
    "num_epochs": 1
}
