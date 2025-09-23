from typing import List

import numpy as np
import nibabel
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import labeled_comprehension


def radii_statistics(
    bin_airway: np.ndarray,
    skeleton: np.ndarray,
    skeleton_parse: np.ndarray,
    spacing: List,
):

    bin_airway = bin_airway.T
    skeleton = skeleton.T

    distance_field = distance_transform_edt(bin_airway, sampling=spacing)

    radii = distance_field * skeleton

    labels = np.unique(skeleton_parse)
    labels = labels[labels > 0]

    max_values = labeled_comprehension(
        radii, skeleton_parse, labels, np.max, float, 0
    )

    min_values = labeled_comprehension(
        radii, skeleton_parse, labels, np.min, float, 0
    )

    mean_values = labeled_comprehension(
        radii, skeleton_parse, labels, np.mean, float, 0
    )

    std_values = labeled_comprehension(
        radii, skeleton_parse, labels, np.std, float, 0
    )
    result = np.column_stack((max_values, min_values, mean_values, std_values))
    return result

