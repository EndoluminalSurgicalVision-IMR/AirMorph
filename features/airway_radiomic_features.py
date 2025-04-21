# Airway radiomic feature extractor
import json
import multiprocessing.connection
import time
import logging
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from collections import OrderedDict
from os.path import join
from ast import literal_eval
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai.transforms import Transform
from radiomics.featureextractor import RadiomicsFeatureExtractor

from util.utils import load_itk_image


class AirwaySignRadiomicFeatures(Transform):
    """ Monai-styled Transform for airway radiomic feature extraction. """

    def __init__(self):

        # Logger
        self.logger = logging.getLogger("RadiomicFeature")
        logging.getLogger("radiomics").setLevel(logging.ERROR)

        self.logger.info("Radiomic Feature Extractor initialized.")

    def __call__(self, data: Dict[str, Any]):

        # ------ Loading necessary components ------
        self.file_path = data.get("file_path")
        self.patient = data.get("patient")
        label_annots: str = data.get("annotations")
        params: str = data.get("params")
        if not label_annots.endswith(".json"):
            self.logger.error("Label mapping file should be of type JSON.")
            return data
        if not params.endswith((".yaml", ".yml")):
            self.logger.error("Params file should be of type YAML.")
            return data

        with open(label_annots, "r") as f:
            self.mapping: Dict[str, Dict[str, str]] = json.load(f)
        self.logger.info("Label mapping file loaded!")
        with open(params, "r") as f:
            self.params: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)
        self.logger.info("Params file loaded!")
        self.logger.info(f"Current radiomic params: {self.params}")
        # ------ Loading necessary components ------

        # For lobar, segmental and sub-segmental, perform feature extraction.
        for level in ["lob", "seg", "subseg"]:
            start_t = time.time()
            features = self.radiomic_feature_extraction(level)
            print("Time used: ", time.time() - start_t)

            # Organize extracted features into csv table
            img_types: List[str] = list(self.params["imageType"].keys())
            type_str = "_".join([s.lower() for s in img_types])
            csv_name = f"airway_{level}_{type_str}_radiomics_features.csv"

            df = pd.DataFrame(features)
            df.insert(0, "annotation", list(self.mapping[level].values()))
            df.to_csv(join(self.file_path, csv_name), sep=",", index=False)
            self.logger.info(f"Radiomics for {level} saved!")

    def radiomic_feature_extraction(self, level: str) -> List[OrderedDict]:
        """ Main code for radiomic feature extraction. """

        if level not in ["lob", "seg", "subseg"]:
            self.logger.error("level must be one of: 'lob', 'seg', 'subseg'.")
            return

        anno_d = self.mapping[level]
        # NOTE: 0 in ITK is used as background, so we need to plus 1.
        # Trachea is omitted
        full_labels = [literal_eval(k) + 1 for k in anno_d.keys()]
        omit_labels = [literal_eval(k) + 1 for k, v in anno_d.items()
                       if v in ["trachea", ]]
        # Based on current anatomical level, load airway classification.
        suffix = level if level != "subseg" else "sub"
        # Read airway classification is of shape [D, H, W]
        airway_cls, origin, spacing, direction = load_itk_image(
            join(self.file_path, "%s_pred_%s.nii.gz" % (self.patient, suffix)))
        current_labels = self._get_class_labels(airway_cls)
        current_labels.remove(0)  # Remove label for background in ITK

        ct_img = load_itk_image(join(self.file_path, "image.nii.gz"))[0]
        lunglobe = load_itk_image(join(self.file_path, "lunglobe.nii.gz"))[0]
        # Ensure correct dtype
        ct_img = ct_img.astype(np.float32)
        airway_cls = airway_cls.astype(np.int32)

        # Remap HU values and crop the airway classification
        ct_img = AirwaySignRadiomicFeatures.cutoff_remap_HU(
            ct_img, -1000, 400, 0, 255)
        ct_img, airway_cls = AirwaySignRadiomicFeatures.crop_lunglobe(
            ct_img, airway_cls, lunglobe)

        shape = ct_img.shape  # Use the shape after cropping
        image_meta = [origin[::-1], direction[::-1], spacing[::-1]]

        # Create a SharedMemory object across all processes. (python >= 3.8)
        features: Dict[OrderedDict] = []
        # NOTE: Change maximum parallel process count based on your own
        # configuration (memory, number of CPU cores, etc.)
        maxParallelProcess = 16
        SHARED_LABEL = SharedMemory("airway_cls", True, airway_cls.nbytes)
        SHARED_CT = SharedMemory("ct_img", True, ct_img.nbytes)
        try:
            labelMemView = np.ndarray(shape, np.int32, buffer=SHARED_LABEL.buf)
            ctMemView = np.ndarray(shape, np.float32, buffer=SHARED_CT.buf)
            labelMemView[:] = airway_cls  # Writing to underlying buffers
            ctMemView[:] = ct_img  # Writing to underlying buffers
            self.logger.warning(f"Shared memory for {level.upper()} created!")

            # Parallel here
            context = multiprocessing.get_context("spawn")
            self.procs = {}  # Maintain all processes
            self.parent_ends = {}  # Maintain parent-end of all pipes
            for tar_label in full_labels:
                if tar_label not in current_labels or tar_label in omit_labels:
                    self.logger.warning(f"Missing {tar_label}.")
                    features.append(OrderedDict([("label", tar_label - 1)]))
                    continue

                # Parent end of the pipe recieves extracted features
                self.parent_ends[tar_label], child_end = context.Pipe()
                self.procs[tar_label] = context.Process(
                    target=AirwaySignRadiomicFeatures.worker,
                    args=(tar_label, child_end, shape, self.params, image_meta))
                self.procs[tar_label].start()

                if len(self.procs) >= maxParallelProcess:
                    # Join all processes
                    for lbl, proc in self.procs.items():
                        proc.join()
                    self.procs.clear()

            # Aggregate results and clear up all processes
            for lbl in self.parent_ends.keys():
                result = self.parent_ends[lbl].recv()  # block
                features.append(result)
                self.parent_ends[lbl].close()

            features = sorted(features, key=lambda x: x["label"])
        finally:
            SHARED_CT.close()
            SHARED_CT.unlink()
            SHARED_LABEL.close()
            SHARED_LABEL.unlink()

        return features

    @staticmethod
    def worker(target_label, pipe, shape, params, image_meta):
        """ Radiomic Extraction worker. """

        SHARED_CT = SharedMemory("ct_img")  # Reference to the mem
        SHARED_LABEL = SharedMemory("airway_cls")  # Reference to the mem
        try:
            ct_img = np.ndarray(shape, np.float32, buffer=SHARED_CT.buf)
            label = np.ndarray(shape, np.int32, buffer=SHARED_LABEL.buf)

            # Create bounding box ROI from target label.
            coords = np.argwhere(label == target_label).reshape(-1, 3)
            z_min, y_min, x_min = np.amin(coords, axis=0, keepdims=False)
            z_max, y_max, x_max = np.amax(coords, axis=0, keepdims=False)
            bbox = np.zeros_like(label, dtype=np.uint8)
            bbox[z_min: z_max + 1,
                 y_min: y_max + 1,
                 x_min: x_max + 1] = 1

            # Transform to ITK image, as input to radiomics feature extractor
            roi = AirwaySignRadiomicFeatures.get_itk_image(bbox, *image_meta)
            ct = AirwaySignRadiomicFeatures.get_itk_image(ct_img, *image_meta)

            # Radiomics extraction process
            extractor = RadiomicsFeatureExtractor(params)
            features = OrderedDict([("label", target_label - 1)])
            radiomics: OrderedDict[str, Any] = extractor.execute(ct, roi, 1)
            features |= radiomics  # Update features

            # Send results through pipe, and close memory reference
            pipe.send(features)
        finally:
            SHARED_CT.close()
            SHARED_LABEL.close()

    @staticmethod
    def get_itk_image(arr: np.ndarray, origin, direction, spacing):
        """ Create a sitk.Image object from array. """

        itk_image = sitk.GetImageFromArray(arr)
        itk_image.SetSpacing(spacing)
        itk_image.SetOrigin(origin)
        itk_image.SetDirection(direction)

        return itk_image

    @staticmethod
    def cutoff_remap_HU(arr: np.ndarray, a_min, a_max, b_min, b_max):
        """ Cutoff based on [a_min, a_max], then remaps HU values to
        [b_min, b_max]. Operations are performed IN-PLACE. """

        # First perform cut-off operation.
        arr[arr <= a_min] = a_min
        arr[arr >= a_max] = a_max

        # Then perform HU scaling.
        arr = b_min + (b_max - b_min) * (arr - a_min) / (a_max - a_min)

        return arr

    @staticmethod
    def crop_lunglobe(ct_img: np.ndarray, label: np.ndarray, lobe: np.ndarray):
        """ Crop both CT image and classification label image, based on lung
        lobe region. """

        # Get the boundaries of lung lobe.
        if not ct_img.shape == lobe.shape or not label.shape == lobe.shape:
            raise ValueError(f"Mismatched shapes: Lunglobe: {lobe.shape}"
                             f" CT: {ct_img.shape} label: {label.shape}")

        lobe_coords = np.argwhere(lobe).reshape(-1, 3)
        z_min, y_min, x_min = np.amin(lobe_coords, axis=0, keepdims=False)
        z_max, y_max, x_max = np.amax(lobe_coords, axis=0, keepdims=False)

        ct_crop = ct_img[
            z_min: z_max + 1, y_min: y_max + 1, x_min: x_max + 1].copy()
        label_crop = label[
            z_min: z_max + 1, y_min: y_max + 1, x_min: x_max + 1].copy()

        return ct_crop, label_crop

    def _get_class_labels(self, label: np.ndarray) -> List[int]:
        """ Return a list of available classes in current segment label. """

        return sorted(np.unique(label).squeeze().tolist())
