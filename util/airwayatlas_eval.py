# -*- coding: utf-8 -*-


import numpy as np
from skimage.morphology import skeletonize
import kimimaro
from scipy import ndimage

__all__ = [
    'dice_coefficient_score_calculation',
    'tree_length_calculation',
    'branch_detected_calculation',
    'iou_score_calculation',
    'cl_score',
    'clDice',
    'compute_skeleton_physical_length',
    'compute_detected_branch_num'
]


def compute_detected_branch_num(skeleton: np.ndarray, thresh: int = 1) -> int:
    """

    Args:
        skeleton:
        thresh: use 1 to surpass the pseudo branchpoints.

    Returns:

    """
    skeleton_indicator = skeleton.copy(
    )  # use 1 for potential branchpoint, 0 for normal skeleton
    neighbor_kernel = ndimage.generate_binary_structure(3, 3)
    skeleton_convolved = ndimage.convolve(skeleton, neighbor_kernel) * skeleton
    skeleton_indicator[skeleton_convolved > 3] = 0
    nd_labels, num = ndimage.label(
        skeleton_indicator, structure=neighbor_kernel)
    sizes = np.bincount(nd_labels.ravel())
    mask = sizes > thresh
    mask[0] = False
    filtered_skeleton = mask[nd_labels]
    _, num_final = ndimage.label(filtered_skeleton, structure=neighbor_kernel)
    return num_final


def compute_skeleton_physical_length(skeleton_points, voxel_spacing, neighbor_mode='26'):
    """
    计算 skeleton 的物理长度（避免重复计算）

    :param skeleton_points: (N, 3) 形状的 NumPy 数组，骨架点的体素坐标
    :param voxel_spacing: (s_x, s_y, s_z) 体素间距 (mm)
    :param neighbor_mode: '6' (六邻域), '18' (十八邻域), '26' (二十六邻域)
    :return: skeleton 物理长度 (mm)
    """
    skeleton_set = {tuple(p) for p in skeleton_points}
    voxel_spacing = np.array(voxel_spacing)

    # 定义邻域搜索偏移
    if neighbor_mode == '6':
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                   (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif neighbor_mode == '18':
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                   (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                   (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
                   (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)]
    elif neighbor_mode == '26':
        offsets = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if
                   not (dx == dy == dz == 0)]

    total_length = 0.0
    visited = set()

    # 遍历骨架点，查找邻居
    for point in skeleton_points:
        point_tuple = tuple(point)
        for offset in offsets:
            neighbor = tuple(np.array(point) + np.array(offset))
            if neighbor in skeleton_set:
                # 确保 (A, B) 和 (B, A) 不重复计算
                edge = tuple(sorted([point_tuple, neighbor]))
                if edge not in visited:
                    visited.add(edge)
                    diff = (np.array(neighbor) - np.array(point)) * \
                        voxel_spacing
                    total_length += np.linalg.norm(diff)

    return total_length


def cl_score(*, s_skeleton: np.array, v_image: np.array) -> float:
    """[this function computes the skeleton volume overlap]
    Args:
        s ([bool]): [skeleton]
        v ([bool]): [image]
    Returns:
        [float]: [computed skeleton volume intersection]

    meanings of v, s refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """
    if np.sum(s_skeleton) == 0:
        return 0
    return float(np.sum(s_skeleton * v_image) / np.sum(s_skeleton))


def convert_multiclass_to_binary(array: np.array) -> np.array:
    """merge all non-background labels into binary class for clDice"""
    return np.where(array > 0, True, False)


def clDice(*, v_p_pred: np.array, v_l_gt: np.array) -> float:
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]

    meanings of v_l, v_p, s_l, s_p refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """

    # NOTE: skeletonization works on binary images;
    # need to convert multiclass to binary mask first
    pred_mask = convert_multiclass_to_binary(v_p_pred)
    gt_mask = convert_multiclass_to_binary(v_l_gt)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize
    if len(pred_mask.shape) == 2:
        # tprec: Topology Precision
        tprec = cl_score(s_skeleton=skeletonize(pred_mask), v_image=gt_mask)
        # tsens: Topology Sensitivity
        tsens = cl_score(s_skeleton=skeletonize(gt_mask), v_image=pred_mask)
    elif len(pred_mask.shape) == 3:
        # tprec: Topology Precision
        tprec = cl_score(s_skeleton=skeletonize(
            pred_mask, method="lee"), v_image=gt_mask)
        # tsens: Topology Sensitivity
        tsens = cl_score(s_skeleton=skeletonize(
            gt_mask, method="lee`"), v_image=pred_mask)

    if (tprec + tsens) == 0:
        return 0

    cldice = 2 * tprec * tsens * 100 / (tprec + tsens)

    return round(cldice, 2)


def calculate_teasar_centerline_for_multiclass_aorta(label):
    TEASAR_SCALE = 1.8
    TEASAR_CONST = 4
    TEASAR_PDRF_SCALE = 100000
    TEASAR_PDRF_EXPONENT = 8
    TEASAR_MAX_PATHS = 10000
    TEASAR_SOMAACCEPTANCE_THRESHOLD = 10000
    TEASAR_DETECTION_THRESHOLD = 10000
    TEASAR_INVALIDATION_CONST = 300
    TEASAR_INVALIDATION_SCALE = 2
    FIX_BRANCHING = True  # default true

    # label_filled, N = fill_voids.fill(label, return_fill_count=True)
    # print('N:{}'.format(N))

    skels = kimimaro.skeletonize(
        label,
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
        dust_threshold=200,
        fix_branching=FIX_BRANCHING,
        fix_borders=True,
        fill_holes=False,
        fix_avocados=False,
        progress=False,
        parallel=1,
        parallel_chunk_size=100,
    )

    vertices = None
    # print('filename:{} skeleton time taken:{} s'.format(file, time.time() - start_time))
    for skel_id, skel in skels.items():
        vertices = np.array(skel.vertices)  # 顶点坐标数组

    vertices = vertices.astype(int)
    skeleton_array = np.zeros(shape=label.shape, dtype=np.uint8)
    skeleton_array[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = 1

    return skeleton_array


def clDice_TEASAR(*, v_p_pred: np.array, v_l_gt: np.array) -> float:
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]

    meanings of v_l, v_p, s_l, s_p refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """

    # NOTE: skeletonization works on binary images;
    # need to convert multiclass to binary mask first
    pred_mask = convert_multiclass_to_binary(v_p_pred)
    gt_mask = convert_multiclass_to_binary(v_l_gt)

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=calculate_teasar_centerline_for_multiclass_aorta(
        pred_mask), v_image=gt_mask)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=calculate_teasar_centerline_for_multiclass_aorta(
        gt_mask), v_image=pred_mask)

    if (tprec + tsens) == 0:
        return 0

    cldice = 2 * tprec * tsens * 100 / (tprec + tsens)

    return round(cldice, 2)


def branch_detected_calculation(pred, label_parsing, label_skeleton, thresh=0.8):
    label_branch = label_skeleton * label_parsing
    label_branch_flat = label_branch.flatten()
    label_branch_bincount = np.bincount(label_branch_flat)[1:]
    total_branch_num = label_branch_bincount.shape[0]
    pred_branch = label_branch * pred
    pred_branch_flat = pred_branch.flatten()
    pred_branch_bincount = np.bincount(pred_branch_flat)[1:]
    if total_branch_num != pred_branch_bincount.shape[0]:
        lack_num = total_branch_num - pred_branch_bincount.shape[0]
        pred_branch_bincount = np.concatenate(
            (pred_branch_bincount, np.zeros(lack_num)))
    branch_ratio_array = pred_branch_bincount / label_branch_bincount
    branch_ratio_array = np.where(branch_ratio_array >= thresh, 1, 0)
    detected_branch_num = np.count_nonzero(branch_ratio_array)
    detected_branch_ratio = round(
        (detected_branch_num * 100) / total_branch_num, 2)
    return total_branch_num, detected_branch_num, detected_branch_ratio


def dice_coefficient_score_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    intersection = np.sum(pred * label)
    dice_coefficient_score = round(
        ((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)) * 100, 2)
    return dice_coefficient_score


def iou_score_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    intersection = np.sum(pred * label)
    union = np.sum(np.logical_or(pred, label))
    iou_score = round((intersection / (union + smooth)) * 100, 2)
    return iou_score


def tree_length_calculation(pred, label_skeleton, smooth=1e-5):
    pred = pred.flatten()
    label_skeleton = label_skeleton.flatten()
    tree_length = round((np.sum(pred * label_skeleton) + smooth) /
                        (np.sum(label_skeleton) + smooth) * 100, 2)
    return tree_length


def false_positive_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fp = np.sum(pred - pred * label) + smooth
    fpr = round(fp * 100 / (np.sum((1.0 - label)) + smooth), 3)
    return fpr


def false_negative_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fn = np.sum(label - pred * label) + smooth
    fnr = round(fn * 100 / (np.sum(label) + smooth), 3)
    return fnr


def sensitivity_calculation(pred, label):
    sensitivity = round(100 - false_negative_rate_calculation(pred, label), 3)
    return sensitivity


def specificity_calculation(pred, label):
    specificity = round(100 - false_positive_rate_calculation(pred, label), 3)
    return specificity


def precision_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    tp = np.sum(pred * label) + smooth
    precision = round(tp * 100 / (np.sum(pred) + smooth), 3)
    return precision
