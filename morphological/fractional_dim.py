# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats


def trim_volume(volume: np.ndarray):
    coords = np.array(np.nonzero(volume))
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1
    return volume[min_coords[0]:max_coords[0],
                  min_coords[1]:max_coords[1],
                  min_coords[2]:max_coords[2]]


def boxcount_3d(Z: np.ndarray, k: int):
    shape = np.array(Z.shape)
    pad_sizes = [(0, (k - s % k) % k) for s in shape]
    Z_padded = np.pad(Z, pad_sizes, mode='constant', constant_values=0)

    new_shape = Z_padded.shape
    shape_reshaped = (new_shape[0] // k, k,
                      new_shape[1] // k, k, new_shape[2] // k, k)
    Z_reshaped = Z_padded.reshape(shape_reshaped)

    box_presence = np.any(Z_reshaped, axis=(1, 3, 5))
    return np.sum(box_presence)


def pad_to_512(arr: np.ndarray, target_shape=(512, 512, 512), pad_value=0):
    original_shape = arr.shape
    padded_arr = np.full(target_shape, pad_value, dtype=arr.dtype)
    start_indices = [(t - o) // 2 for o,
                     t in zip(original_shape, target_shape)]
    slices = tuple(slice(start, start + o)
                   for start, o in zip(start_indices, original_shape))
    padded_arr[slices] = arr
    return padded_arr


def fractal_dimension_3d(Z: np.ndarray, min_box_size=1, max_box_size=7):
    p = min(Z.shape)
    if max_box_size is None:
        exponents = np.arange(np.floor(np.log2(p)), 0, -1, dtype=int)
    else:
        exponents = np.arange(max_box_size, 0, -1, dtype=int)
    sizes = 2 ** exponents
    sizes = sizes[::-1]

    counts = []
    for k in sizes:
        count = boxcount_3d(Z, k)
        counts.append(count)

    log_sizes = np.log(1.0 / sizes)  # log(1/box size)
    log_counts = np.log(counts)
    slope, _, _, _, _ = stats.linregress(
        log_sizes, log_counts)
    fractal_dim = slope

    return fractal_dim


def compute_fd_branch(
    skeleton: np.ndarray,
    airway_cls: np.ndarray,
    index: int,
) -> float:
    skeleton = skeleton.T
    airway_cls = airway_cls.T

    skel_cls = skeleton * airway_cls
    skel_cls = np.where(skel_cls == index, 1, 0)
    if skel_cls.sum() > 0:
        skel_cls = trim_volume(skel_cls)
        skel_cls = pad_to_512(skel_cls)
        fd = fractal_dimension_3d(skel_cls, max_box_size=8)
    else:
        fd = 0.0

    return fd
