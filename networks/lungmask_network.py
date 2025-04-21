# -*- coding: utf-8 -*-


import logging
import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)
print(project_root)

import warnings
from typing import Optional, Union
from typing import Tuple

import SimpleITK as sitk
import fill_voids
import numpy as np
import pydicom as pyd
import skimage
import skimage.measure
import skimage.morphology
import torch
import torch.nn.functional as F

from more_itertools import chunked
from scipy import ndimage
from torch import nn
from tqdm import tqdm


def preprocess(
        img: np.ndarray, resolution: list = [192, 192]
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses the image by clipping, cropping and resizing. Clipping at -1024 and 600 HU, cropping to the body

    Args:
        img (np.ndarray): Image to be preprocessed
        resolution (list, optional): Target size after preprocessing. Defaults to [192, 192].

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed image and the cropping bounding box
    """
    imgmtx = np.copy(img)
    imgmtx = np.clip(imgmtx, -1024, 600)
    cip_xnew = []
    cip_box = []
    for imslice in imgmtx:
        im, box = crop_and_resize(
            imslice, width=resolution[0], height=resolution[1])
        cip_xnew.append(im)
        cip_box.append(box)
    return np.asarray(cip_xnew), cip_box


def simple_bodymask(img: np.ndarray) -> np.ndarray:
    """Computes a simple bodymask by thresholding the image at -500 HU and then filling holes and removing small objects

    Args:
        img (np.ndarray): CT image (single slice) in HU

    Returns:
        np.ndarray: Binary mask of the body
    """

    # Here are some heuristics to get a body mask
    maskthreshold = -500
    oshape = img.shape
    img = ndimage.zoom(img, 128 / np.asarray(img.shape), order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(
        int
    )
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape) / 128
    return ndimage.zoom(bodymask, real_scaling, order=0)


def crop_and_resize(
        img: np.ndarray, width: int = 192, height: int = 192
) -> Tuple[np.ndarray, np.ndarray]:
    """Crops the image to the body and resizes it to the specified size

    Args:
        img (np.ndarray): Image to be cropped and resized
        width (int, optional): Target width to be resized to. Defaults to 192.
        height (int, optional): Target height to be resized to. Defaults to 192.

    Returns:
        Tuple[np.ndarray, np.ndarray]: resized image and the cropping bounding box
    """
    bmask = simple_bodymask(img)
    # img[bmask==0] = -1024 # this line removes background outside of the lung.
    # However, it has been shown problematic with narrow circular field of views that touch the lung.
    # Possibly doing more harm than help
    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    if len(reg) > 0:
        bbox = np.asarray(reg[0].bbox)
    else:
        bbox = (0, 0, bmask.shape[0], bmask.shape[1])
    img = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]
    img = ndimage.zoom(
        img, np.asarray([width, height]) / np.asarray(img.shape), order=1
    )
    return img, bbox


def reshape_mask(mask: np.ndarray, tbox: np.ndarray, origsize: tuple) -> np.ndarray:
    """Reshapes the mask to the original size given bounding box and original size

    Args:
        mask (np.ndarray): Mask to be resampled (nearest neighbor)
        tbox (np.ndarray): Bounding box in original image covering field of view of the mask
        origsize (tuple): Original images size

    Returns:
        np.ndarray: Resampled mask in original image space
    """
    res = np.ones(origsize) * 0
    resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
    imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
    res[tbox[0]: tbox[2], tbox[1]: tbox[3]] = imgres
    return res


def read_dicoms(path, primary=True, original=True, disable_tqdm=False):
    allfnames = []
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]

    dcm_header_info = []
    unique_set = (
        []
    )  # need this because too often there are duplicates of dicom files with different names
    i = 0
    for fname in tqdm(allfnames, disable=disable_tqdm):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        i += 1
        if filename_[0] != "DICOMDIR":
            try:
                dicom_header = pyd.dcmread(
                    fname, defer_size=100, stop_before_pixels=True, force=True
                )
                if dicom_header is not None:
                    if "ImageType" in dicom_header:
                        if primary:
                            is_primary = all(
                                [x in dicom_header.ImageType for x in ["PRIMARY"]]
                            )
                        else:
                            is_primary = True

                        if original:
                            is_original = all(
                                [x in dicom_header.ImageType for x in ["ORIGINAL"]]
                            )
                        else:
                            is_original = True

                        if (
                                is_primary
                                and is_original
                                and "LOCALIZER" not in dicom_header.ImageType
                        ):
                            h_info_wo_name = [
                                dicom_header.StudyInstanceUID,
                                dicom_header.SeriesInstanceUID,
                                dicom_header.ImagePositionPatient,
                            ]
                            h_info = [
                                dicom_header.StudyInstanceUID,
                                dicom_header.SeriesInstanceUID,
                                fname,
                                dicom_header.ImagePositionPatient,
                            ]
                            if h_info_wo_name not in unique_set:
                                unique_set.append(h_info_wo_name)
                                dcm_header_info.append(h_info)

            except Exception as e:
                logging.error("Unexpected error:", e)
                logging.warning(
                    "Doesn't seem to be DICOM, will be skipped: ", fname)

    conc = [x[1] for x in dcm_header_info]
    sidx = np.argsort(conc)
    conc = np.asarray(conc)[sidx]
    dcm_header_info = np.asarray(dcm_header_info, dtype=object)[sidx]
    vol_unique = np.unique(conc, return_index=1,
                           return_inverse=1)  # unique volumes
    n_vol = len(vol_unique[1])
    if n_vol == 1:
        logging.info("There is " + str(n_vol) + " volume in the study")
    else:
        logging.info("There are " + str(n_vol) + " volumes in the study")

    relevant_series = []
    relevant_volumes = []

    for i in range(len(vol_unique[1])):
        curr_vol = i
        info_idxs = np.where(vol_unique[2] == curr_vol)[0]
        vol_files = dcm_header_info[info_idxs, 2]
        positions = np.asarray(
            [np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]]
        )
        slicesort_idx = np.argsort(positions)
        vol_files = vol_files[slicesort_idx]
        relevant_series.append(vol_files)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(vol_files)
        vol = reader.Execute()
        relevant_volumes.append(vol)

    return relevant_volumes


def load_input_image(path: str, disable_tqdm=False) -> sitk.Image:
    """Loads image, if path points to a file, file will be loaded. If path points ot a folder, a DICOM series will be loaded. If multiple series are present, the largest series (higher number of slices) will be loaded.

    Args:
        path (str): File or folderpath to be loaded. If folder, DICOM series is expected
        disable_tqdm (bool, optional): Disable tqdm progress bar. Defaults to False.

    Returns:
        sitk.Image: Loaded image
    """
    if os.path.isfile(path):
        logging.info(f"Read input: {path}")
        input_image = sitk.ReadImage(path)
    else:
        logging.info(f"Looking for dicoms in {path}")
        dicom_vols = read_dicoms(
            path, original=False, primary=False, disable_tqdm=disable_tqdm
        )
        if len(dicom_vols) < 1:
            sys.exit("No dicoms found!")
        if len(dicom_vols) > 1:
            logging.warning(
                "There are more than one volume in the path, will take the largest one"
            )
        input_image = dicom_vols[
            np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)
        ]
    return input_image


def postprocessing(
        label_image: np.ndarray,
        spare: list = [],
        disable_tqdm: bool = False,
        skip_below: int = 3,
) -> np.ndarray:
    """some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. Only largest connected components (CC) for each label will be kept. If a label is member of the spare list it will be mapped to neighboring labels and not present in the final labelling.

    Args:
        label_image (np.ndarray): Label image (int) to be processed
        spare (list, optional): Labels that are used for mapping to neighbors but not considered for final labelling. This is used for label fusion with a filling model. Defaults to [].
        disable_tqdm (bool, optional): If true, tqdm will be diabled. Defaults to False.
        skip_below (int, optional): If a CC is smaller than this value. It will not be merged but removed. This is for performance optimization.

    Returns:
        np.ndarray: Postprocessed volume
    """

    # CC analysis
    regionmask = skimage.measure.label(label_image)
    origlabels = np.unique(label_image)
    origlabels_maxsub = np.zeros(
        (max(origlabels) + 1,), dtype=np.uint32
    )  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        r_max_intensity = int(r.max_intensity)
        if r.area > origlabels_maxsub[r_max_intensity]:
            origlabels_maxsub[r_max_intensity] = r.area
            region_to_lobemap[r.label] = r_max_intensity

    for r in tqdm(regions, disable=disable_tqdm):
        r_max_intensity = int(r.max_intensity)
        if (
                r.area < origlabels_maxsub[r_max_intensity] or r_max_intensity in spare
        ) and r.area >= skip_below:  # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0]: bb[1], bb[2]: bb[3], bb[4]: bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n not in spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto

            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if (
                    regions[regionlabels.index(mapto)].area
                    == origlabels_maxsub[
                        int(regions[regionlabels.index(mapto)].max_intensity)
                    ]
            ):
                origlabels_maxsub[
                    int(regions[regionlabels.index(mapto)].max_intensity)
                ] += myarea
            regions[regionlabels.index(mapto)].__dict__[
                "_cache"]["area"] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[np.isin(outmask_mapped, spare)] = 0

    if outmask_mapped.shape[0] == 1:
        holefiller = (
            lambda x: skimage.morphology.area_closing(
                x[0].astype(int), area_threshold=64
            )[None, :, :]
            == 1
        )
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(
            outmask_mapped == i))] = i

    return outmask


def bbox_3D(labelmap, margin=2):
    """Compute bounding box of a 3D labelmap.

    Args:
        labelmap (np.ndarray): Input labelmap
        margin (int, optional): Margin to add to the bounding box. Defaults to 2.

    Returns:
        np.ndarray: Bounding box as [zmin, zmax, ymin, ymax, xmin, xmax]
    """
    shape = labelmap.shape
    dimensions = np.arange(len(shape))
    bmins = []
    bmaxs = []
    margin = [margin] * len(dimensions)
    for dim, dim_margin, dim_shape in zip(dimensions, margin, shape):
        margin_label = np.any(labelmap, axis=tuple(
            dimensions[dimensions != dim]))
        bmin, bmax = np.where(margin_label)[0][[0, -1]]
        bmin -= dim_margin
        bmax += dim_margin + 1
        bmin = max(bmin, 0)
        bmax = min(bmax, dim_shape)
        bmins.append(bmin)
        bmaxs.append(bmax)

    bbox = np.array(list(zip(bmins, bmaxs))).flatten()
    return bbox


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keeps largest connected component (CC)

    Args:
        mask (np.ndarray): Input label map

    Returns:
        np.ndarray: Binary label map with largest CC
    """
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv', residual=False):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
            residual: if True, residual connections will be added
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if i == 0 and residual:
                self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                    padding, batch_norm, residual, first=True))
            else:
                self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                    padding, batch_norm, residual))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode,
                                            padding, batch_norm, residual))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        res = self.last(x)
        return self.softmax(res)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, residual=False, first=False):
        super(UNetConvBlock, self).__init__()
        self.residual = residual
        self.out_size = out_size
        self.in_size = in_size
        self.batch_norm = batch_norm
        self.first = first
        self.residual_input_conv = nn.Conv2d(
            self.in_size, self.out_size, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm2d(self.out_size)

        if residual:
            padding = 1
        block = []

        if residual and not first:
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(in_size))

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))

        if not residual:
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            if self.in_size != self.out_size:
                x = self.residual_input_conv(x)
                x = self.residual_batchnorm(x)
            out = out + x

        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, residual=False):
        super(UNetUpBlock, self).__init__()
        self.residual = residual
        self.in_size = in_size
        self.out_size = out_size
        self.residual_input_conv = nn.Conv2d(
            self.in_size, self.out_size, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm2d(self.out_size)

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    @staticmethod
    def center_crop(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out_orig = torch.cat([up, crop1], 1)
        out = self.conv_block(out_orig)
        if self.residual:
            if self.in_size != self.out_size:
                out_orig = self.residual_input_conv(out_orig)
                out_orig = self.residual_batchnorm(out_orig)
            out = out + out_orig

        return out


warnings.filterwarnings("ignore", category=UserWarning)

# stores urls and number of classes of the models
MODEL_URLS = {
    "R231": (
        "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth",
        3,
    ),
    "LTRCLobes": (
        "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth",
        6,
    ),
    "R231CovidWeb": (
        "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231covid-0de78a7e.pth",
        3,
    ),
}


def get_model(
        modelname: str, modelpath: Optional[str] = None, n_classes: int = 3
) -> torch.nn.Module:
    """Loads specific model and state

    Args:
        modelname (str): Modelname (e.g. R231, LTRCLobes or R231CovidWeb)
        modelpath (Optional[str], optional): Path to statedict, if not provided will be downloaded automatically. Modelname will be ignored if provided. Defaults to None.
        n_classes (int, optional): Number of classes. Will be automatically set if modelname is provided. Defaults to 3.

    Returns:
        torch.nn.Module: Loaded model in eval state
    """
    if modelpath is None:
        model_url, n_classes = MODEL_URLS[modelname]
        state_dict = torch.hub.load_state_dict_from_url(
            model_url, progress=True, map_location=torch.device("cpu")
        )
    else:
        state_dict = torch.load(modelpath, map_location=torch.device("cpu"))

    model = UNet(
        n_classes=n_classes,
        padding=True,
        depth=5,
        up_mode="upsample",
        batch_norm=True,
        residual=False,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


class LMInferer:
    def __init__(
            self,
            modelname="R231",
            fillmodel: Optional[str] = None,
            force_cpu=False,
            batch_size=20,
            volume_postprocessing=True,
            noHU=False,
            tqdm_disable=False,
    ):
        """LungMaskInference

        Args:
            modelname (str, optional): Model to be applied. Defaults to 'R231'.
            fillmodel (Optional[str], optional): Fillmodel to be applied. Defaults to None.
            force_cpu (bool, optional): Will not use GPU is `True`. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 20.
            volume_postprocessing (bool, optional): If `Fales` will not perform postprocessing (connected component analysis). Defaults to True.
            noHU (bool, optional): If `True` no HU intensities are expected. Not recommended. Defaults to False.
            tqdm_disable (bool, optional): If `True`, will disable progress bar. Defaults to False.
        """
        assert (
            modelname in MODEL_URLS
        ), "Modelname not found. Please choose from: {}".format(MODEL_URLS.keys())
        if fillmodel is not None:
            assert (
                fillmodel in MODEL_URLS
            ), "Modelname not found. Please choose from: {}".format(MODEL_URLS.keys())
        self.fillmodel = fillmodel
        self.modelname = modelname
        self.force_cpu = force_cpu
        self.batch_size = batch_size
        self.volume_postprocessing = volume_postprocessing
        self.noHU = noHU
        self.tqdm_disable = tqdm_disable

        modelpath = os.path.join(project_root, r'checkpoints/unet_r231-d5d2fc3d.pth')
        
        self.model = get_model(self.modelname,
                               modelpath=modelpath)

        self.device = torch.device("cpu")
        if not self.force_cpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                logging.info("No GPU found, using CPU instead")
        self.model.to(self.device)

        self.fillmodelm = None
        if self.fillmodel is not None:
            self.fillmodelm = get_model(self.fillmodel)
            self.fillmodelm.to(self.device)

    def _inference(
            self, image: Union[sitk.Image, np.ndarray], model: torch.nn.Module
    ) -> np.ndarray:
        """Performs model inference

        Args:
            image (Union[sitk.Image, np.ndarray]): Input image (volumetric)
            model (torch.nn.Module): Model to be applied

        Returns:
            np.ndarray: Inference result
        """
        numpy_mode = isinstance(image, np.ndarray)
        if numpy_mode:
            inimg_raw = image.copy()
        else:
            curr_orient = (
                sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                    image.GetDirection()
                )
            )
            if curr_orient != "LPS":
                image = sitk.DICOMOrient(image, "LPS")
            inimg_raw = sitk.GetArrayFromImage(image)

        if self.noHU:
            # support for non HU images. This is just a hack. The models were not trained with this in mind
            tvolslices = skimage.color.rgb2gray(inimg_raw)
            tvolslices = skimage.transform.resize(tvolslices, [256, 256])
            tvolslices = np.asarray(
                [tvolslices * x for x in np.linspace(0.3, 2, 20)])
            tvolslices[tvolslices > 1] = 1
            sanity = [
                (tvolslices[x] > 0.6).sum() > 25000 for x in range(len(tvolslices))
            ]
            tvolslices = tvolslices[sanity]
        else:
            tvolslices, xnew_box = preprocess(inimg_raw, resolution=[256, 256])
            tvolslices[tvolslices > 600] = 600
            tvolslices = np.divide((tvolslices + 1024), 1624)

        timage_res = np.empty(
            (np.append(0, tvolslices[0].shape)), dtype=np.uint8)

        with torch.no_grad():
            for mbnp in tqdm(
                    chunked(tvolslices, self.batch_size),
                    disable=self.tqdm_disable,
                    total=len(tvolslices) / self.batch_size,
            ):
                mbt = torch.as_tensor(
                    np.asarray(mbnp)[:, None, ::],
                    dtype=torch.float32,
                    device=self.device,
                )
                prediction = model(mbt)
                pred = (
                    torch.max(prediction, 1)[1].detach(
                    ).cpu().numpy().astype(np.uint8)
                )
                timage_res = np.vstack((timage_res, pred))

        # postprocessing includes removal of small connected components, hole filling and mapping of small components to
        # neighbors
        if self.volume_postprocessing:
            outmask = postprocessing(
                timage_res, disable_tqdm=self.tqdm_disable)
        else:
            outmask = timage_res

        if self.noHU:
            outmask = skimage.transform.resize(
                outmask[np.argmax((outmask == 1).sum(axis=(1, 2)))],
                inimg_raw.shape[:2],
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            )[None, :, :]
        else:
            outmask = np.asarray(
                [
                    reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:])
                    for i in range(outmask.shape[0])
                ],
                dtype=np.uint8,
            )

        if not numpy_mode:
            if curr_orient != "LPS":
                outmask = sitk.GetImageFromArray(outmask)
                outmask = sitk.DICOMOrient(outmask, curr_orient)
                outmask = sitk.GetArrayFromImage(outmask)

        return outmask.astype(np.uint8)

    def apply(self, image: Union[sitk.Image, np.ndarray]) -> np.ndarray:
        """Apply model on image (volumetric)

        Args:
            image (Union[sitk.Image, np.ndarray]): Input image

        Returns:
            np.ndarray: Lung segmentation
        """
        if self.fillmodel is None:
            return self._inference(image, self.model)
        else:
            logging.info(f"Apply: {self.modelname}")
            res_l = self._inference(image, self.model)
            logging.info(f"Apply: {self.fillmodel}")
            res_r = self._inference(image, self.fillmodelm)
            spare_value = res_l.max() + 1
            res_l[np.logical_and(res_l == 0, res_r > 0)] = spare_value
            res_l[res_r == 0] = 0
            logging.info(
                "Fusing results... this may take up to several minutes!")
            return postprocessing(res_l, spare=[spare_value])


def apply(
        image: Union[sitk.Image, np.ndarray],
        model=None,
        force_cpu=False,
        batch_size=20,
        volume_postprocessing=True,
        noHU=False,
        tqdm_disable=False,
):
    inferer = LMInferer(
        force_cpu=force_cpu,
        batch_size=batch_size,
        volume_postprocessing=volume_postprocessing,
        noHU=noHU,
        tqdm_disable=tqdm_disable,
    )
    if model is not None:
        inferer.model = model.to(inferer.device)
    return inferer.apply(image)


def apply_fused(
        image,
        basemodel="LTRCLobes",
        fillmodel="R231",
        force_cpu=False,
        batch_size=20,
        volume_postprocessing=True,
        noHU=False,
        tqdm_disable=False,
):
    inferer = LMInferer(
        modelname=basemodel,
        force_cpu=force_cpu,
        fillmodel=fillmodel,
        batch_size=batch_size,
        volume_postprocessing=volume_postprocessing,
        noHU=noHU,
        tqdm_disable=tqdm_disable,
    )
    return inferer.apply(image)
