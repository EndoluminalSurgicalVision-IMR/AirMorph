
## for airway cls model

import numpy as np
import math
import json

import os
import sys

left = np.array([11,3,4,5,6,8,9,10])
def Class2Anno(classes):
    Anno = np.zeros((classes.shape[0],3))
    for i in range(classes.shape[0]):
        class_i = classes[i]
        if 1<= class_i<= 56:
            Anno[i,0] = 1
            Anno[i,1] = left[math.ceil(class_i/7) -1]
            Anno[i,2] = (class_i - 1)%7
        if 57 <= class_i <= 126:
            Anno[i, 0] = 2
            Anno[i, 1] = math.ceil((class_i - 56)/7)
            Anno[i, 2] = (class_i - 1)%7
        if 126 < class_i :
            Anno[i, 0] = 1
            Anno[i, 1] = 7
            Anno[i, 2] = class_i -127
    return Anno

def calculate_CS(pred_seg, pred_sub, mask_top,label_old2new_dictpath):
    label_old2new = json.load(open(label_old2new_dictpath))
    label_new2old = {}
    for k, v in label_old2new.items():
        label_new2old[int(v)] = int(k)
    pred_sub = np.array([label_new2old[k] for k in pred_sub])

    top_seg = np.zeros(mask_top.shape[0])
    cal_seg = np.zeros(mask_top.shape[0])
    for i in range(mask_top.shape[0]):
        if pred_seg[i] == 18:
            continue
        if pred_seg[mask_top[i, :] == 1].shape[0] > 1:
            cal_seg[i] = 1
            if np.all(pred_seg[mask_top[i, :] == 1] == pred_seg[i]):
                top_seg[i] = 1
    Anno = Class2Anno(pred_sub)
    top_sub = np.zeros(pred_sub.shape[0])
    cal_sub = np.zeros(pred_sub.shape[0])
    for i in range(pred_sub.shape[0]):
        if pred_sub[i] == 0:
            continue
        if pred_sub[mask_top[i, :] == 1].shape[0] > 1:
            cal_sub[i] = 1
            if np.all(Anno[mask_top[i, :] == 1, 0] == Anno[i, 0]) and np.all(
                    Anno[mask_top[i, :] == 1, 1] == Anno[i, 1]):
                if 1 <= Anno[i, 2] <= 3:
                    if np.all(Anno[mask_top[i, :] == 1, 2] == Anno[i, 2]):
                        top_sub[i] = 1
                if Anno[i, 2] == 0:
                    top_sub[i] = 1
                if Anno[i, 2] == 4:
                    if np.all((Anno[mask_top[i, :] == 1, 2] == 1) | (Anno[mask_top[i, :] == 1, 2] == 2)):
                        top_sub[i] = 1
                if Anno[i, 2] == 5:
                    if np.all((Anno[mask_top[i, :] == 1, 2] == 2) | (Anno[mask_top[i, :] == 1, 2] == 3)):
                        top_sub[i] = 1
                if Anno[i, 2] == 6:
                    if np.all((Anno[mask_top[i, :] == 1, 2] == 1) | (Anno[mask_top[i, :] == 1, 2] == 3)):
                        top_sub[i] = 1
    return np.sum(top_seg) / np.sum(cal_seg), np.sum(top_sub) / np.sum(cal_sub)


