import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)

sys.path.append(project_root)


import logging
import warnings
import traceback

from monai.transforms import Compose

from segmentator.airway_segmentator import AirwayAtlasBinaryAirwaySegmentator
from classifier.airway_classifier import AirwayAtlasMultiAnatomyAirwayClassifier


if __name__ == "__main__":

    # os.system("clear")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    # Initialize pycuda and make context
    pipelines = Compose([
        AirwayAtlasBinaryAirwaySegmentator(),
        AirwayAtlasMultiAnatomyAirwayClassifier(),
    ])

    data = {
        "patient": "080",
        "file_path": os.path.join(project_root, r"sample_data/ATM22/080")
    }
    try:
        data = pipelines(data)

    except Exception:
        traceback.print_exc()
