# Class for binary airway segmentation.
import os
import logging
from typing import Dict

import torch
import numpy as np
from monai.transforms import Transform

from util.utils import (
    load_itk_image,
    save_itk,
    crop_image_via_box,
    restore_image_via_box,
    InnerTransformer,
)
from util.airwayatlas_utils import (
    IMRAirwayAtlas_SkeletonExtraction,
    IMRAirwayAtlas_SkeletonExtraction_HighRecallLowPrecision,  # noqa
    IMRAirwayAtlas_SkeletonExtraction_LowRecallHighPrecision,  # noqa
)
from models.airway_model import AirwayExtractionModel
from models.airway_model_wingsnet import AirwayExtractionModel_wingsnet
from models.airway_model_break import AirwayExtractionModel_break
from models.lungmask_model import LungMaskExtractionModel


class AirwayAtlasBinaryAirwaySegmentator(Transform):
    """ Monai-styled Transform for extraction from CT scans the following:
     - Lobe
     - Ensembled binary airway prediction
     - Binary airway skeleton
    """

    def __init__(self):

        # Logger
        self.logger = logging.getLogger("BinaryAirwaySeg")

        # Binary segmentation models
        self.lobeextractor = LungMaskExtractionModel()
        self.airwayextractor = AirwayExtractionModel()
        self.airwayextractor_break = AirwayExtractionModel_break()
        self.airwayextractor_wingsnet = AirwayExtractionModel_wingsnet()

        self.logger.info("Binary Airway Segmentator initialized.")

    def _extract_lobe(self, image: np.ndarray):
        """ Lobe extraction. Crop the input CT image based on Lobe bounding box
        . """

        lobe = self.lobeextractor.predict(image)
        lunglobe_bbox = self.lobeextractor.lung_boundingbox
        image_crop: np.ndarray = crop_image_via_box(image, lunglobe_bbox)

        self.logger.info("Lobe extraction complete.")

        return lobe, image_crop, lunglobe_bbox

    def _extract_airway(self, image: np.ndarray):
        """ Ensembled airway extraction. """
        airway1, airway2, airway3 = self.airwayextractor.predict(image)
        airway_wingsnet = self.airwayextractor_wingsnet.predict(image)
        airway_break = self.airwayextractor_break.predict(image)
        airway_ensemble = airway1 + airway2 + airway3 + \
            airway_wingsnet + airway_break
        airway_ensemble[airway_ensemble < 3] = 0
        airway_ensemble[airway_ensemble >= 3] = 1

        pred_ensemble_lcc = airway_ensemble
        pred_ensemble_lcc = InnerTransformer.KeepLargestConnectedComponent(
            pred_ensemble_lcc)
        pred_ensemble_lcc = InnerTransformer.ToNumpy(pred_ensemble_lcc)
        pred_ensemble_lcc = InnerTransformer.CastToNumpyUINT8(
            pred_ensemble_lcc[0, ...])

        torch.cuda.empty_cache()
        self.logger.info("Airway segmentation complete.")
        return pred_ensemble_lcc

    def __call__(self, data: Dict[str, str], **kwds) -> Dict[str, np.ndarray]:

        file_path: str | None = data.get("file_path", None)
        if file_path is None:
            raise ValueError("No data path provided.")

        image, origin, spacing, direction = load_itk_image(
            os.path.join(file_path, "image.nii.gz"))
        lobe, image_crop, lobe_bbox = self._extract_lobe(image)
        bin_airway_crop = self._extract_airway(image_crop)
        bin_airway = restore_image_via_box(
            image.shape, bin_airway_crop, lobe_bbox)
        skel = IMRAirwayAtlas_SkeletonExtraction(bin_airway, spacing)
        self.logger.info("Skeleton computation complete.")

        # Save predicted lobe and binary airway
        save_itk(lobe, os.path.join(file_path, "lunglobe.nii.gz"),
                 origin, spacing, direction)
        save_itk(bin_airway, os.path.join(file_path, "airway_bin.nii.gz"),
                 origin, spacing, direction)
        save_itk(skel, os.path.join(file_path, "airway_skeleton.nii.gz"),
                 origin, spacing, direction)
        self.logger.info("Saving complete.")

        data.update({
            "image": image,  # [D, H, W]
            "lobe": lobe,
            "image_crop": image_crop,
            "bin_airway": bin_airway,
            "skeleton": skel,
            "itk_meta": {
                "origin": origin,
                "spacing": spacing,
                "direction": direction,
            },
        })
        return data
