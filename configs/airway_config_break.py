# -*- coding: utf-8 -*-


import os


current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))



config = {
    "in_channels": 1,
    "out_channels": 1,
    "finalsigmoid": 1,
    "fmaps_degree": 16,
    "fmaps_layer_number": 4,
    "layer_order": "cip",
    "GroupNormNumber": 4,
    "device": "cuda:0",
    "weight_path1": os.path.join(project_root, r'checkpoints/break1.ckpt'),
    "roi_size": (128, 224, 304),
    "sw_batch_size": 1,
    "overlap": 0.75,
    "mode": 'gaussian',
    "sigma_scale": 0.25,
    "KeepLargestConnectedComponent": True,
    "use_HU_window": True
}
