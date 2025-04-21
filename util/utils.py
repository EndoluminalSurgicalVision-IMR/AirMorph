# -*- coding: utf-8 -*-

import os
from typing import Tuple, List

import SimpleITK as sitk
import numpy as np

from monai.inferers import sliding_window_inference
from monai.transforms import (
    KeepLargestConnectedComponent,
    ToNumpy,
    AsDiscrete,
    CastToType,
    SqueezeDim,
    ToTensor,
    EnsureChannelFirst,
)
import torch


class InnerTransform(object):

    def __init__(self):

        self.ToNumpy = ToNumpy()
        self.AsDiscrete = AsDiscrete(threshold=0.5)
        self.ArgMax = AsDiscrete(argmax=True)
        self.KeepLargestConnectedComponent = KeepLargestConnectedComponent(
            applied_labels=1, connectivity=3)
        self.EnsureChannelFirst = EnsureChannelFirst()
        self.CastToNumpyUINT8 = CastToType(dtype=np.uint8)
        self.AddChannel = EnsureChannelFirst(channel_dim="no_channel")
        self.SqueezeDim = SqueezeDim()
        self.ToTensor = ToTensor(dtype=torch.float32)


InnerTransformer = InnerTransform()


def save_itk(image, filename, origin, spacing, direction):

    if not isinstance(origin, Tuple):
        if isinstance(origin, List):
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))

    if not isinstance(spacing, Tuple):
        if isinstance(spacing, List):
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))

    if not isinstance(direction, Tuple):
        if isinstance(direction, List):
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))

    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename, True)


def load_itk_image(filename):

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))

    return numpyImage, numpyOrigin, numpySpacing, numpyDirection


#  See
#  D:\N\2_b\pre\1_nodule_voi\tests\_test_all_v2_resample_sitk_ResampleImageFilter_all.py
#  and
#  https://blog.csdn.net/m0_37477175/article/details/90751407
#  and
#  https://blog.csdn.net/weixin_40640335/article/details/115798249
#  https://blog.csdn.net/jancis/article/details/106265602
#  https://www.codeleading.com/article/84462249014/
#  https://blog.csdn.net/qq_26628975/article/details/118388911

def ImageResample(sitk_image: sitk.Image, new_spacing=[1.0, 1.0, 1.0], is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''

    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    # 这样 spacing 还会又更小的小数，但感觉肯能是有必要的
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)  # 设置输出图像大小
    resample.SetOutputSpacing(new_spacing_refine)  # 设置输出图像间距

    if is_label:
        # 根据需要重采样图像的情况设置不同的dype
        # https://blog.csdn.net/weixin_40640335/article/details/115798249
        resample.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
        # 插值过程中，mask图像用最近邻插值，CT图像用线性插值
        # https://blog.csdn.net/m0_37477175/article/details/90751407
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # 线性插值用于PET/CT/MRI之类的，保存float32
        resample.SetOutputPixelType(sitk.sitkFloat32)
        # resample.SetInterpolator(sitk.sitkBSpline)  # 83s
        resample.SetInterpolator(sitk.sitkLinear)  # 1s
        # resample.SetInterpolator(sitk.sitkBSpline1)  # 90s
        # resample.SetInterpolator(sitk.sitkBSpline3)  # 106s
        # 之前是 simpleitk2.0.2，没有sitk.sitkBSpline1；更新到simpleitk2.2.1，才有sitk.sitkBSpline1；但是也要70s

    # resample.SetTransform(sitk.Euler3DTransform())  #  这个和插值没啥关系吧
    newimage = resample.Execute(sitk_image)
    return newimage, new_spacing_refine

    # 有的很多的加了下面的代码，有什么区别吗？
    # 代码一
    # 我猜这个要加在最前面，加了这个就不要SetOutputDirection和SetOutputOrigin了，后面的SetOutputSpacing和SetSize会被覆盖改变
    # resample.SetReferenceImage(sitk_image)  # 需要重新采样的目标图像，将输出的大小、原点、间距和方向设置为itkimage
    # 代码二
    # https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1Transform.html#ace15b81212fdf30ce61af0b03b09a71e
    # 我觉得 由于官方 By default a 3-d identity transform is constructed，默认的可以不写
    # resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    # 有的SetTransform用的是下面的代码（https://blog.csdn.net/qq_26628975/article/details/118388911），有什么区别吗
    # transform = sitk.Transform()
    # transform.SetIdentity()
    # resampler.SetTransform(transform)
    # 有的用的下面这个  https://www.cnblogs.com/Mr-Mango/articles/14398030.html
    # resample.SetTransform(sitk.Euler3DTransform())


# Interpolator 可选：https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
# transform 可选：https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a527cb966ed81d0bdc65999f4d2d4d852

def ImageResampleZ(sitk_image, new_spacing=[1.0, 1.0, 1.0], is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''

    size = np.array(sitk_image.GetSize())
    # print(size)
    spacing = np.array(sitk_image.GetSpacing())
    # print(spacing)
    new_spacing = np.array(new_spacing)
    print(new_spacing)
    new_spacing[0] = spacing[0]
    new_spacing[1] = spacing[1]
    print(new_spacing)

    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    # 这样 spacing 还会又更小的小数，但感觉肯能是有必要的
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)  # 设置输出图像大小
    resample.SetOutputSpacing(new_spacing_refine)  # 设置输出图像间距

    if is_label:
        # 根据需要重采样图像的情况设置不同的dype
        # https://blog.csdn.net/weixin_40640335/article/details/115798249
        resample.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
        # 插值过程中，mask图像用最近邻插值，CT图像用线性插值
        # https://blog.csdn.net/m0_37477175/article/details/90751407
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # 线性插值用于PET/CT/MRI之类的，保存float32
        resample.SetOutputPixelType(sitk.sitkFloat32)
        # resample.SetInterpolator(sitk.sitkBSpline)  # 83s
        resample.SetInterpolator(sitk.sitkLinear)  # 1s
        # resample.SetInterpolator(sitk.sitkBSpline1)  # 90s
        # resample.SetInterpolator(sitk.sitkBSpline3)  # 106s
        # 之前是 simpleitk2.0.2，没有sitk.sitkBSpline1；更新到simpleitk2.2.1，才有sitk.sitkBSpline1；但是也要70s

    # resample.SetTransform(sitk.Euler3DTransform())  #  这个和插值没啥关系吧
    newimage = resample.Execute(sitk_image)
    return newimage, new_spacing_refine


def load_itk_image_with_sampling(filename, spacing=[0.8, 0.8, 0.8]):

    itkimage = sitk.ReadImage(filename)
    new_image_sitk, new_spacing_refine = ImageResample(
        itkimage, new_spacing=spacing, is_label=False)
    numpyImage = sitk.GetArrayFromImage(new_image_sitk)  # z, y, x
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    # numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    # return numpyImage, numpyOrigin, numpySpacing, numpyDirection
    return new_image_sitk, numpyImage, numpyOrigin, list(reversed(new_spacing_refine)), numpyDirection


def crop_image_via_box(image, box):

    return image[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1], box[2, 0]:box[2, 1]]


def restore_image_via_box(origin_shape, image, box):

    # np.uint8 is default
    origin_image = np.zeros(shape=origin_shape, dtype=np.uint8)
    origin_image[box[0, 0]:box[0, 1],
                 box[1, 0]:box[1, 1],
                 box[2, 0]:box[2, 1]] = image
    return origin_image


def restore_likelihood_via_box(origin_shape, image, box):

    # np.float32 for the likelihood
    origin_image = np.zeros(shape=origin_shape, dtype=np.float16)
    origin_image[box[0, 0]:box[0, 1],
                 box[1, 0]:box[1, 1],
                 box[2, 0]:box[2, 1]] = image
    return origin_image


def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)


def dcm2nii(dcms_path, nii_path):

    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    sitk.WriteImage(image2, nii_path)
