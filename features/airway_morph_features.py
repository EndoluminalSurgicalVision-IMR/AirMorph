# Airway morphological feature extractor
import os
import logging
from typing import Dict, Any, List

import numpy as np
from monai.transforms import Transform

from util.utils import load_itk_image
from morphological.angle_scope import compute_max_open_angle
from morphological.length import analyze_label_K_paths
from morphological.twist import process_all_branches_with_pca
from morphological.radii import radii_statistics
from morphological.fractional_dim import compute_fd_branch


class AirwaySignMorphologicalFeatures(Transform):
    """ Monai-styled Transform for airway morphological feature extraction. """

    NUM_LOB_CLS = 5
    NUM_SEG_CLS = 18

    def __init__(self):

        # Logger
        self.logger = logging.getLogger("MorphFeature")

        self.logger.info("Morphological Feature Extractor initialized.")

    def find_bbox_3D(self, mask: np.ndarray):
        """ Return bounding box of 3D segmentation """

        if len(mask.shape) != 3:
            raise ValueError(
                f"The dimension of input is {len(mask.shape)}, not 3.")

        pos = np.where(mask)
        return pos[0].min(), pos[0].max() + 1, \
            pos[1].min(), pos[1].max() + 1, pos[2].min(), pos[2].max() + 1

    def __call__(self, data: Dict[str, Any]):

        patient = data.get("patient")

        # ------ Loading necessary components ------
        file_path = data.get("file_path")
        bin_airway, _, spacing, _ = load_itk_image(
            os.path.join(file_path, "airway_bin.nii.gz"))  # [D, H, W]
        spacing = spacing[::-1]  # [x, y, z] order
        skeleton, _, _, _ = load_itk_image(
            os.path.join(file_path, "airway_skeleton.nii.gz"))  # [D, H, W]
        x = np.load(os.path.join(
            file_path, "%s_airway_feature_cls.npy" % patient))
        edge = np.load(os.path.join(
            file_path, "%s_airway_graph.npy" % patient))
        y_lob, y_seg, _ = np.load(os.path.join(
            file_path, "%s_airway_graph_cls.npy" % patient))
        skel_parse, _, _, _ = load_itk_image(
            os.path.join(file_path, "%s_skel_parsing.nii.gz" % patient))
        skel_parse = skel_parse.T
        airway_parse, _, _, _ = load_itk_image(
            os.path.join(file_path, "%s_parse.nii.gz" % patient))
        airway_parse = airway_parse.T
        airway_seg_cls, _, _, _ = load_itk_image(
            os.path.join(file_path, "%s_pred_seg.nii.gz" % patient))
        airway_lob_cls, _, _, _ = load_itk_image(
            os.path.join(file_path, "%s_pred_lob.nii.gz" % patient))

        self.logger.info("Loading complete.")
        # ------ Loading necessary components ------

        # Compute length of trachea for further analysis
        trachea_ind = np.argmin(x[:, 0]) + 1
        trachea_skel = (skel_parse == trachea_ind).astype(np.uint8)
        x_min, x_max, y_min, y_max, z_min, z_max = self.find_bbox_3D(
            trachea_skel)
        delx = abs(x_max - x_min) * spacing[0]
        dely = abs(y_max - y_min) * spacing[1]
        delz = abs(z_max - z_min) * spacing[2]
        trachea_len = np.sqrt(delx ** 2 + dely ** 2 + delz ** 2)

        # Compute Divergence (maximum open angle) at lobar and segmental level
        div_seg = np.array(
            [compute_max_open_angle(edge, x[:, 0], y_seg, x[:, 1:4], j)
             for j in range(self.NUM_SEG_CLS)])
        div_lob = np.array([
            compute_max_open_angle(edge, x[:, 0], y_lob, x[:, 1:4], j + 1)
            for j in range(self.NUM_LOB_CLS)])
        self.logger.info("Divergence computed!")

        # Compute Length at lobar and segmental level
        len_seg = np.array(
            [analyze_label_K_paths(
                edge, x[:, 0], y_seg, x[:, 4], j) * trachea_len
             for j in range(self.NUM_SEG_CLS)])
        len_seg[len_seg < 0] = -1  # Normalization
        len_lob = np.array(
            [analyze_label_K_paths(
                edge, x[:, 0], y_lob, x[:, 4], j + 1) * trachea_len
             for j in range(self.NUM_LOB_CLS)])
        len_lob[len_lob < 0] = -1  # Normalization
        self.logger.info("Length computed!")

        # Compute Complexity at lobar and segmental level
        com_seg = np.array([
            compute_fd_branch(skeleton.copy(), airway_seg_cls.copy(), j)
            for j in range(1, self.NUM_SEG_CLS + 1)])
        com_lob = np.array([
            compute_fd_branch(skeleton.copy(), airway_lob_cls.copy(), j)
            for j in range(2, self.NUM_LOB_CLS + 2)])
        self.logger.info("Complexity computed!")

        # Compute Tortuosity at lobar and segmental level
        tor_branch = process_all_branches_with_pca(airway_parse, spacing)
        tor_seg = np.array(
            [tor_branch[y_seg == j, 1].mean() / np.pi
             if np.any(y_seg == j) else -1
             for j in range(self.NUM_SEG_CLS)])
        tor_lob = np.array(
            [tor_branch[y_lob == (j + 1), 1].mean() / np.pi
             if np.any(y_lob == (j + 1)) else -1
             for j in range(self.NUM_LOB_CLS)])
        self.logger.info("Tortuosity computed!")

        # Compute Stenosis at lobar and segmental level
        radii = radii_statistics(
            bin_airway.copy(), skeleton.copy(), skel_parse.copy(), spacing)
        ste_seg = np.array([
            np.mean(1 - (radii[y_seg == j, 1] / radii[y_seg == j, 2]))
            if np.any(y_seg == j) else -1
            for j in range(self.NUM_SEG_CLS)
        ])
        ste_lob = np.array([
            np.mean(1 - (radii[y_lob == (j + 1), 1] /
                    radii[y_lob == (j + 1), 2]))
            if np.any(y_lob == (j + 1)) else -1
            for j in range(self.NUM_LOB_CLS)
        ])
        self.logger.info("Stenosis computed!")

        # Compute Ectasia at lobar and segmental level
        # TODO: Change ectasia computation from mean/max to max/mean
        ect_seg = np.array([
            np.mean(radii[y_seg == j, 2] / radii[y_seg == j, 0])
            if np.any(y_seg == j) else -1
            for j in range(self.NUM_SEG_CLS)
        ])
        ect_lob = np.array([
            np.mean(radii[y_lob == (j + 1), 2] / radii[y_lob == (j + 1), 0])
            if np.any(y_lob == (j + 1)) else -1
            for j in range(self.NUM_LOB_CLS)
        ])
        self.logger.info("Ectasia computed!")

        # Stack all features as morphological signature
        mor_lob = np.column_stack(
            (div_lob, len_lob, com_lob, tor_lob, ste_lob, ect_lob))
        mor_seg = np.column_stack(
            (div_seg, len_seg, com_seg, tor_seg, ste_seg, ect_seg))
        morph_features = np.row_stack((mor_lob, mor_seg))

        # Save data
        np.save(os.path.join(file_path, "%s_morphological_features.npy" % patient),
                morph_features)
        self.logger.info("Morphological features saved!")

        # Update data dict for radiomic feature extraction
        data.update({
            "morphological": morph_features,
        })
        return data
