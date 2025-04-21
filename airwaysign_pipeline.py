import os
import logging
import traceback
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from monai.transforms import Compose

from features.airway_morph_features import AirwaySignMorphologicalFeatures
from features.airway_radiomic_features import AirwaySignRadiomicFeatures


if __name__ == "__main__":

    os.system("clear")

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    pipelines = Compose([
        AirwaySignMorphologicalFeatures(),
        # AirwaySignRadiomicFeatures(),
    ])
    data = {
        "patient": "080",
        "file_path": os.path.join(project_root,"sample_data/ATM22/080"),
        "annotations": os.path.join(project_root,"configs/class2anno.json"),
        "params": os.path.join(project_root,"configs/radiomic_params.yaml"),
    }
    try:
        data = pipelines(data)

    except Exception:
        traceback.print_exc()
