# -*- coding: utf-8 -*-



from typing import List, Tuple, Dict, Callable, Union, Any
import sys
import os
import SimpleITK as sitk
import numpy as np

NDArray = np.ndarray

__all__ = [
    'load_nii',
    'save_nii',
]


def load_nii(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    return numpyImage, (numpyOrigin, numpySpacing, numpyDirection)


# def save_nii(image, filename, origin, spacing, direction):
#     if type(origin) != tuple:
#         if type(origin) == list:
#             origin = tuple(reversed(origin))
#         else:
#             origin = tuple(reversed(origin.tolist()))
#     if type(spacing) != tuple:
#         if type(spacing) == list:
#             spacing = tuple(reversed(spacing))
#         else:
#             spacing = tuple(reversed(spacing.tolist()))
#     if type(direction) != tuple:
#         if type(direction) == list:
#             direction = tuple(reversed(direction))
#         else:
#             direction = tuple(reversed(direction.tolist()))
#     itkimage = sitk.GetImageFromArray(image, isVector=False)
#     itkimage.SetSpacing(spacing)
#     itkimage.SetOrigin(origin)
#     itkimage.SetDirection(direction)
#     sitk.WriteImage(itkimage, filename, True)

def save_nii(image, filename, meta_info):

    if not isinstance(meta_info[0], Tuple):
        if isinstance(meta_info[0], List):
            origin = tuple(reversed(meta_info[0]))
        else:
            origin = tuple(reversed(meta_info[0].tolist()))

    if not isinstance(meta_info[1], Tuple):
        if isinstance(meta_info[1], List):
            spacing = tuple(reversed(meta_info[1]))
        else:
            spacing = tuple(reversed(meta_info[1].tolist()))

    if not isinstance(meta_info[2], Tuple):
        if isinstance(meta_info[2], List):
            direction = tuple(reversed(meta_info[2]))
        else:
            direction = tuple(reversed(meta_info[2].tolist()))

    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename, True)
