# -*- coding: utf-8 -*-
import os
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))


config = {
    "device": "cuda:0",
    "weight_paths": [os.path.join(project_root, r'checkpoints/airway_cls.ckpt')],
    "label_old2new_dictpath": os.path.join(project_root,'models/label_old2new.json'),
    "input_dim": 11,
    "num_classes1": 6,
    "num_classes2": 20,
    "num_classes3": 82,
    "dim": 128,
    "heads": 4,
    "mlp_dim": 256,
    "dim_head": 32,
    "dropout": 0.,
    "trans_depth": 2,
    "outlier_depth": 2,
}
