# -*- coding: utf-8 -*-

from networks.airway_network_break import WingsNet_v3, normalize_CT, lumTrans
from configs.airway_config_break import config
from util.utils import InnerTransformer, sliding_window_inference

import torch
import numpy as np

from skimage.morphology import skeletonize
import skimage.measure as measure
from scipy import ndimage


class AirwayExtractionModel_break(object):
    def __init__(self):
        self.config = config
        self.device = self.config['device']
        self.net = WingsNet_v3()  
        self.net = self.net.to(self.device) 
    @torch.no_grad()
    def predict(self, image: np.ndarray):
        self.net.eval()
        if self.config['use_HU_window']:
            image = lumTrans(image)
        image = normalize_CT(image)
        image = InnerTransformer.ToTensor(image)
        image = InnerTransformer.AddChannel(image)
        image = InnerTransformer.AddChannel(image)
        image = image.to(self.device)
        self.net.load_state_dict(
            torch.load(self.config['weight_path1'], map_location=lambda storage, loc: storage.cuda(0))[
                "state_dict"]) 
        pred2 = sliding_window_inference(
            inputs=image,
            roi_size=self.config['roi_size'],
            sw_batch_size=self.config['sw_batch_size'],
            predictor=self.net,
            overlap=self.config['overlap'],
            mode=self.config['mode'],
            sigma_scale=self.config['sigma_scale']
        )
        pred = pred2
        pred = InnerTransformer.AsDiscrete(
            pred[:, 0, ...]) 
        return pred
