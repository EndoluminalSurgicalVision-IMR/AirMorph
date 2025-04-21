# -*- coding: utf-8 -*-



import kimimaro
import numpy as np

from typing import List, Tuple, Dict, Callable, Union, Any

__all__ = [
    'IMRAirwayAtlas_SkeletonExtraction',
    'IMRAirwayAtlas_SkeletonExtraction_HighRecallLowPrecision',
    'IMRAirwayAtlas_SkeletonExtraction_LowRecallHighPrecision',
]


# from .volume_utils import load_nii, save_nii


def IMRAirwayAtlas_SkeletonExtraction(vol: np.ndarray,
                                      spacing: np.ndarray) \
        -> np.ndarray:
    """

    Use path-cost way to build the skeleton of the airway used in IMR AirwayAtlas,
        for the following graph-representation build.

    Args:
        vol: 3D-array.
        spacing: spacing of the vol,

    Returns:
        skel: 3D-array.

    """

    vol = vol.astype(np.uint8)
    max_voxel_reso = np.max(spacing)
    # parameters
    TEASAR_SCALE = 5 * max_voxel_reso
    TEASAR_CONST = 3 * max_voxel_reso
    TEASAR_PDRF_SCALE = 100000
    TEASAR_PDRF_EXPONENT = 2
    TEASAR_MAX_PATHS = 10000
    TEASAR_SOMAACCEPTANCE_THRESHOLD = 10000
    TEASAR_DETECTION_THRESHOLD = 10000
    TEASAR_INVALIDATION_CONST = 300
    TEASAR_INVALIDATION_SCALE = 2
    FIX_BRANCHING = True

    try:
        skels = kimimaro.skeletonize(
            vol,
            teasar_params={
                "scale": TEASAR_SCALE,
                "const": TEASAR_CONST,
                "pdrf_scale": TEASAR_PDRF_SCALE,
                "pdrf_exponent": TEASAR_PDRF_EXPONENT,
                "soma_acceptance_threshold": TEASAR_SOMAACCEPTANCE_THRESHOLD,
                "soma_detection_threshold": TEASAR_DETECTION_THRESHOLD,
                "soma_invalidation_const": TEASAR_INVALIDATION_CONST,
                "soma_invalidation_scale": TEASAR_INVALIDATION_SCALE,
                "max_paths": TEASAR_MAX_PATHS,
            },
            dust_threshold=0,
            fix_branching=FIX_BRANCHING,
            fix_borders=True,
            fill_holes=False,
            fix_avocados=False,
            progress=True,
            parallel=1,
            parallel_chunk_size=100,
        )
        for skel_id, skel in skels.items():
            vertices = np.array(skel.vertices)

        vertices = vertices.astype(int)
        skeleton_array = np.zeros(shape=vol.shape, dtype=np.uint8)
        skeleton_array[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = 1

        return skeleton_array

    except Exception:
        print('skeleton extract not successfully!')


def IMRAirwayAtlas_SkeletonExtraction_HighRecallLowPrecision(vol: np.ndarray,
                                                             spacing: np.ndarray) \
        -> np.ndarray:
    """

    Ablation Study of the path-cost skeleton extraction, high recall, low precision.

    Args:
        vol: 3D-array.
        spacing: spacing of the vol.

    Returns:
        skel: 3D-array.

    """

    vol = vol.astype(np.uint8)
    # max_voxel_reso = np.max(spacing)
    # parameters
    TEASAR_SCALE = 1.5  # high recall, low precision
    TEASAR_CONST = 1.5  # high recall, low precision
    TEASAR_PDRF_SCALE = 100000
    TEASAR_PDRF_EXPONENT = 2
    TEASAR_MAX_PATHS = 10000
    TEASAR_SOMAACCEPTANCE_THRESHOLD = 10000
    TEASAR_DETECTION_THRESHOLD = 10000
    TEASAR_INVALIDATION_CONST = 300
    TEASAR_INVALIDATION_SCALE = 2
    FIX_BRANCHING = True

    try:
        skels = kimimaro.skeletonize(
            vol,
            teasar_params={
                "scale": TEASAR_SCALE,
                "const": TEASAR_CONST,
                "pdrf_scale": TEASAR_PDRF_SCALE,
                "pdrf_exponent": TEASAR_PDRF_EXPONENT,
                "soma_acceptance_threshold": TEASAR_SOMAACCEPTANCE_THRESHOLD,
                "soma_detection_threshold": TEASAR_DETECTION_THRESHOLD,
                "soma_invalidation_const": TEASAR_INVALIDATION_CONST,
                "soma_invalidation_scale": TEASAR_INVALIDATION_SCALE,
                "max_paths": TEASAR_MAX_PATHS,
            },
            dust_threshold=0,
            fix_branching=FIX_BRANCHING,
            fix_borders=True,
            fill_holes=False,
            fix_avocados=False,
            progress=True,
            parallel=1,
            parallel_chunk_size=100,
        )
        for skel_id, skel in skels.items():
            vertices = np.array(skel.vertices)

        vertices = vertices.astype(int)
        skeleton_array = np.zeros(shape=vol.shape, dtype=np.uint8)
        skeleton_array[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = 1

        return skeleton_array

    except Exception:
        print('skeleton extract not successfully!')


def IMRAirwayAtlas_SkeletonExtraction_LowRecallHighPrecision(vol: np.ndarray,
                                                             spacing: np.ndarray) \
        -> np.ndarray:
    """

    Ablation Study of the path-cost skeleton extraction, low recall, high precision.

    Args:
        vol: 3D-array.
        spacing: spacing of the vol.

    Returns:
        skel: 3D-array.

    """

    vol = vol.astype(np.uint8)
    # max_voxel_reso = np.max(spacing)
    # parameters
    TEASAR_SCALE = 8  # low recall, high precision
    TEASAR_CONST = 8  # low recall, high precision
    TEASAR_PDRF_SCALE = 100000
    TEASAR_PDRF_EXPONENT = 2
    TEASAR_MAX_PATHS = 10000
    TEASAR_SOMAACCEPTANCE_THRESHOLD = 10000
    TEASAR_DETECTION_THRESHOLD = 10000
    TEASAR_INVALIDATION_CONST = 300
    TEASAR_INVALIDATION_SCALE = 2
    FIX_BRANCHING = True

    try:
        skels = kimimaro.skeletonize(
            vol,
            teasar_params={
                "scale": TEASAR_SCALE,
                "const": TEASAR_CONST,
                "pdrf_scale": TEASAR_PDRF_SCALE,
                "pdrf_exponent": TEASAR_PDRF_EXPONENT,
                "soma_acceptance_threshold": TEASAR_SOMAACCEPTANCE_THRESHOLD,
                "soma_detection_threshold": TEASAR_DETECTION_THRESHOLD,
                "soma_invalidation_const": TEASAR_INVALIDATION_CONST,
                "soma_invalidation_scale": TEASAR_INVALIDATION_SCALE,
                "max_paths": TEASAR_MAX_PATHS,
            },
            dust_threshold=0,
            fix_branching=FIX_BRANCHING,
            fix_borders=True,
            fill_holes=False,
            fix_avocados=False,
            progress=True,
            parallel=1,
            parallel_chunk_size=100,
        )
        for skel_id, skel in skels.items():
            vertices = np.array(skel.vertices)

        vertices = vertices.astype(int)
        skeleton_array = np.zeros(shape=vol.shape, dtype=np.uint8)
        skeleton_array[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = 1

        return skeleton_array

    except Exception:
        print('skeleton extract not successfully!')
