# -*- coding: utf-8 -*-

'''
Program :
Author  :   Minghui Zhang, sjtu
File    :   lungmask_model.py
Date    :   2023/6/7 18:38
Version :   V1.0
'''

from networks.lungmask_network import LMInferer
from configs.lungmask_config import config
import SimpleITK as sitk
import torch
import numpy as np


class LungMaskExtractionModel(object):
    def __init__(self):
        self.config = config
        self.device = self.config['device']
        self.net = LMInferer()

        self.lung_boundingbox = None

    @torch.no_grad()
    def predict(self, image):
        pred = self.net.apply(image)
        if self.config["CalculateLungBoundingbox"]:
            self._locate_lung_boundingbox(pred)
        return pred

    def _locate_lung_boundingbox(self, image: np.ndarray):
        xx, yy, zz = np.where(image)
        self.lung_boundingbox = np.array(
            [[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        margin = self.config['margin_lung_boundingbox']
        self.lung_boundingbox = np.vstack([np.max([[0, 0, 0], self.lung_boundingbox[:, 0] - margin], 0),
                                           np.min([np.array(image.shape), self.lung_boundingbox[:, 1] + margin],
                                                  axis=0).T]).T
        return
