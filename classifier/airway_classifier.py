# Class for multi-class airway anatomy classification
import os
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))


import json
import time
import logging
from typing import Dict, Any

import numpy as np
from monai.transforms import Transform

from util.utils import save_itk
from models.labeling_model import AirwaylabelingModel
from data_process.tree_parse_skelbased import tree_parse
from data_process.feature_cuda import feature_extraction_cuda
from data_process.dataset import multitask_dataset
from data_process.data_save_cuda import transform_pred_data


class AirwayAtlasMultiAnatomyAirwayClassifier(Transform):
    """ Monai-styled Transform for airway anatomy multi-class classification.
    """

    def __init__(self):

        # Logger
        self.logger = logging.getLogger("AirwayAnatomyCls")

        # Airway Classification model
        self.model = AirwaylabelingModel()
        self.logger.info("Multi-class Airway Anatomy Classifier initialized.")

    def __call__(self, data: Dict[str, Any]):

        patient: str = data.get("patient")
        bin_airway: np.ndarray = data.get("bin_airway")
        lobe: np.ndarray = data.get("lobe")
        skeleton: np.ndarray = data.get("skeleton")
        itk_meta: Dict = data.get("itk_meta")
        file_path: str = data.get("file_path")

        parent_map, children_map, generation, trachea_ind, tree_parsing_skel, airway_parse = \
            tree_parse(bin_airway.copy(), skeleton.copy())
        self.logger.info("Finished tree-parsing.")

        skel_for_feature = tree_parsing_skel.copy()[::-1, ::-1, ::-1]
        lobe_for_feature = lobe.copy().T[::-1, ::-1, ::-1]

        data_final, edge, edge_feature, node_idx, _ = \
            feature_extraction_cuda(
                skel_for_feature, itk_meta["spacing"][::-1], lobe_for_feature,
                parent_map, children_map, generation, trachea_ind)
        edge_graph = edge[:, edge_feature > 0]
        self.logger.info("Finished feature extraction.")


        dataset = multitask_dataset(
            patient, data_final, edge, edge_feature)
        self.logger.info("Finished inference dataset building.")
 
        start = time.time()
        pred_lob, pred_seg, pred_sub = self.model.predict_select(dataset)
        self.logger.info(f"Finished prediction: {time.time() - start}")

        with open(os.path.join(project_root, r"configs/class2anno.json"), "r")as f:
        # with open("configs/class2anno.json", "r") as f:
            class2anno = json.load(f)
        case_anno = {
            int(node_idx[i] + 1): [
                str(class2anno["lob"][str(pred_lob[i])]),
                str(class2anno["seg"][str(pred_seg[i])]),
                str(class2anno["subseg"][str(pred_sub[i])])
            ]
            for i in range(node_idx.shape[0])
        }

        img_sub = transform_pred_data(airway_parse, node_idx, pred_sub)
        img_seg = transform_pred_data(airway_parse, node_idx, pred_seg)
        img_lob = transform_pred_data(airway_parse, node_idx, pred_lob)
        self.logger.info("Predicted image transformation complete.")


        save_itk(
            tree_parsing_skel.T.astype(np.int32),
            os.path.join(file_path, "%s_skel_parsing.nii.gz" % patient),
            itk_meta["origin"], itk_meta["spacing"], itk_meta["direction"])

        save_itk(
            airway_parse.T.astype(np.int32),
            os.path.join(file_path, "%s_parse.nii.gz" % patient),
            itk_meta["origin"], itk_meta["spacing"], itk_meta["direction"])

        save_itk(
            img_sub.T.astype(np.int32),
            os.path.join(file_path, "%s_pred_sub.nii.gz" % patient),
            itk_meta["origin"], itk_meta["spacing"], itk_meta["direction"])
        save_itk(
            img_seg.T.astype(np.int32),
            os.path.join(file_path, "%s_pred_seg.nii.gz" % patient),
            itk_meta["origin"], itk_meta["spacing"], itk_meta["direction"])
        save_itk(
            img_lob.T.astype(np.int32),
            os.path.join(file_path, "%s_pred_lob.nii.gz" % patient),
            itk_meta["origin"], itk_meta["spacing"], itk_meta["direction"])

        np.save(os.path.join(file_path, "%s_airway_graph.npy" % patient),
                edge_graph)
        np.save(os.path.join(file_path, "%s_airway_feature_cls.npy" % patient),
                data_final)
        np.save(os.path.join(file_path, "%s_airway_graph_cls.npy" % patient),
                [pred_lob, pred_seg, pred_sub])

        with open(os.path.join(file_path, "%s_anno.json" % patient), "w",
                  encoding='utf-8') as f:
            json.dump(case_anno, f, indent=4, ensure_ascii=False)

        self.logger.info("Saving complete.")

        # Update data dict
        data.update({
            "skel_parse": tree_parsing_skel.T.astype(np.int32),  # [D, H, W]
            "airway_parse": airway_parse.T.astype(np.int32),
            "pred_sub": img_sub.copy().T,
            "pred_seg": img_seg.copy().T,
            "pred_lob": img_lob.copy().T,
            "airway_feature_cls": data_final.copy(),
            "airway_graph": edge_graph.copy(),
            "airway_graph_cls": [pred_lob, pred_seg, pred_sub],
        })
        return data
