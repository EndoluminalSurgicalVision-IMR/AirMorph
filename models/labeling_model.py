# -*- coding: utf-8 -*-
import torch
import numpy as np

from models.evaluation_matrics import calculate_CS
from networks.transformer import our_net
from configs.labeling_config import config


class AirwaylabelingModel(object):

    def __init__(self):
        self.config = config
        self.device = [self.config['device']]
        self.net = our_net(
            input_dim=self.config["input_dim"],
            num_classes1=self.config["num_classes1"],
            num_classes2=self.config["num_classes2"],
            num_classes3=self.config["num_classes3"],
            dim=self.config["dim"],
            heads=self.config["heads"],
            mlp_dim=self.config["mlp_dim"],
            dim_head=config["dim_head"],
            dropout=config["dropout"],
            trans_depth=config["trans_depth"],
            outlier_depth=config["outlier_depth"]
        )
        self.label_old2new_dictpath = config["label_old2new_dictpath"]

    def load_checkpoint(self, ckpt_path):

        self.net.load_state_dict(torch.load(ckpt_path)['state_dict'])

    @torch.no_grad()
    def predict(self, case):
        self.net.eval()
        x = case.x
        spd = case.spd
        mask_top = case.mask_top
        _, _, _, output12, output22, output32, _, _, _, _ = self.net(
            x, mask_top, spd.detach(), 0)

        pred12: np.ndarray = output12.max(dim=1)[1].cpu().data.numpy()
        pred22: np.ndarray = output22.max(dim=1)[1].cpu().data.numpy()
        pred32: np.ndarray = output32.max(dim=1)[1].cpu().data.numpy()

        return [pred12, pred22, pred32]

    def predict_select(self, case):
        best_cs_seg = 0
        best_cs_sub = 0
        best_pred_lob = None
        best_pred_seg = None
        best_pred_sub = None
        best_ckpt_seg = None
        best_ckpt_sub = None

        for ckpt_path in self.config['weight_paths']:
            self.load_checkpoint(ckpt_path)
            pred = self.predict(case)
            cs_seg, cs_sub = calculate_CS(
                pred[1], pred[2], case.mask_top.cpu().data.numpy(),
                self.label_old2new_dictpath)
            if cs_seg > best_cs_seg:
                best_cs_seg = cs_seg
                best_pred_seg = pred[1]
                best_pred_lob = pred[0]
                best_ckpt_seg = ckpt_path
            if cs_sub > best_cs_sub:
                best_cs_sub = cs_sub
                best_pred_sub = pred[2]
                best_ckpt_sub = ckpt_path

        print(f"segmental consistency: {best_cs_seg},({best_ckpt_seg}) "
              f"subsegmental consistency: {best_cs_sub}({best_ckpt_sub})")
        return best_pred_lob, best_pred_seg, best_pred_sub
