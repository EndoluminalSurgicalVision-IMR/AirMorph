import random
import os
import sys
from typing import List

import nibabel
import numpy as np
import pycuda.autoprimaryctx
import pycuda.driver as drv
from pycuda.compiler import SourceModule


sys.path.append(os.path.abspath(__file__))


def process_iteration_pycuda(
    skel,
    generation,
    xt,
    yt,
    zt,
    spacing,
    parent_map,
    children_map,
    mask,
    lr
):

    mod = SourceModule("""
    __global__ void process_iteration_kernel(
        float *skel, int* generation,
        float xt, float yt, float zt, float *spacing,
        int* parent_map, int* children_map,
        int* node_idx, float *total, int max_label, int x_dim, int y_dim, int z_dim,
        float* bounding_left, float* bounding_right,
        int* skl1, int* left, int* right, int* skl_p, float *mask, int* skl_left_right) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= max_label) return;
        //if (i != 1) return;
        int idx = i+1;

        int parent = -1;
        for (int j = 0; j < max_label; j++) {
            if (parent_map[(idx - 1) * max_label + j] == 1) {
                parent = j + 1;
                break;
            }
        }
        if (parent == -1) return;

        int sum_skl1 = 0, skl1_left = 0, skl1_right = 0;
        int minX = x_dim, minY = y_dim, minZ = z_dim;
        int maxX = 0, maxY = 0, maxZ = 0;

        int sum_skl_p = 0;
        int minX_skl = x_dim, minY_skl = y_dim, minZ_skl = z_dim;
        int maxX_skl = 0, maxY_skl = 0, maxZ_skl = 0;

        for (int z = 0; z < z_dim; z++) {
            for (int y = 0; y < y_dim; y++) {
                for (int x = 0; x < x_dim; x++) {
                    int k = z + y * z_dim + x * z_dim * y_dim;
                    if(skel[k] == idx){
                        if (mask[k]==2) {
                            atomicExch(&skl_left_right[i], 2);
                        } else if (mask[k]==1) {
                            atomicExch(&skl_left_right[i], 1);
                        }

                        sum_skl1 += 1;
                        // 更新边界值
                        minX = min(minX, x);
                        maxX = max(maxX, x);
                        minY = min(minY, y);
                        maxY = max(maxY, y);
                        minZ = min(minZ, z);
                        maxZ = max(maxZ, z);

                    }

                    if (skel[k] == parent) {
                        skl_p[k] = 1;
                        sum_skl_p += 1;
                        // 更新边界值
                        minX_skl = min(minX_skl, x);
                        maxX_skl = max(maxX_skl, x);
                        minY_skl = min(minY_skl, y);
                        maxY_skl = max(maxY_skl, y);
                        minZ_skl = min(minZ_skl, z);
                        maxZ_skl = max(maxZ_skl, z);
                    }

                }
            }
        }
        if (sum_skl1 == 0) return;
        if (sum_skl_p==0) return;
        int gen = generation[idx - 1] + 1;

        float x = (minX + maxX +1) / 2.0f;
        float y = (minY + maxY +1) / 2.0f;
        float z = (minZ + maxZ +1) / 2.0f;

        float deltax = (x - xt) * spacing[0];
        float deltay = (y - yt) * spacing[1];
        float deltaz = (z - zt) * spacing[2];

        float delx = abs(maxX +1 - minX) * spacing[0];
        float dely = abs(maxY +1 - minY) * spacing[1];
        float delz = abs(maxZ +1 - minZ) * spacing[2];

        float x_re = 0, y_re = 0, z_re = 0;
        if (skl_left_right[i] == 2) {
            x_re = -(x - bounding_left[0]) / (bounding_left[1] - bounding_left[0]);
            y_re = -(y - bounding_left[2]) / (bounding_left[3] - bounding_left[2]);
            z_re = -(z - bounding_left[4]) / (bounding_left[5] - bounding_left[4]);
        } else if (skl_left_right[i] == 1) {
            x_re = (x - bounding_right[0]) / (bounding_right[1] - bounding_right[0]);
            y_re = (y - bounding_right[2]) / (bounding_right[3] - bounding_right[2]);
            z_re = (z - bounding_right[4]) / (bounding_right[5] - bounding_right[4]);
        }

        float length = sqrt(delx * delx + dely * dely + delz * delz);
        //printf("total: %d,%d,%d,%d,\\n", idx, skl1_left, skl1_right, sum_skl1);

        int first_x, first_y, first_z, second_x, second_y, second_z;

        if (abs(minX - minX_skl) < abs(minY - minY_skl)) {
            first_x = minX;
            first_y = minY;
            first_z = minZ;
            second_x = maxX+1;
            second_y = maxY+1;
            second_z = maxZ+1;
        } else {
            first_x = maxX+1;
            first_y = maxY+1;
            first_z = maxZ+1;
            second_x = minX;
            second_y = minY;
            second_z = minZ;
        }

        float delta_x = (second_x - first_x) * spacing[0];
        float delta_y = (first_y - second_y) * spacing[1];
        float delta_z = (first_z - second_z) * spacing[2];
        float d = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);


        float pi = 3.14159265358979323846;
        float cosx = delta_x / d;
        float cosy = delta_y / d;
        float cosz = delta_z / d;
        float theta_x = acos(cosx) / pi * 180.0;
        float theta_y = acos(cosy) / pi * 180.0;
        float theta_z = acos(cosz) / pi * 180.0;

        int children_num = 0;
        for (int j = 0; j < max_label; j++) {
            children_num += children_map[(idx - 1) * max_label + j];
        }

        int brother_num = 0;
        for (int j = 0; j < max_label; j++) {
            brother_num += children_map[(parent - 1) * max_label + j];
        }
        brother_num -= 1;

        float volume = 0;
        for (int k = 0; k < x_dim * y_dim * z_dim; k++) {
            if (skel[k] == idx) {
                volume += 1;
            }
        }
        volume *= spacing[0] * spacing[1] * spacing[2];

        total[(idx - 1) * 17 + 0] = gen;
        total[(idx - 1) * 17 + 1] = deltax;
        total[(idx - 1) * 17 + 2] = deltay;
        total[(idx - 1) * 17 + 3] = deltaz;
        total[(idx - 1) * 17 + 4] = length;
        total[(idx - 1) * 17 + 5] = delx;
        total[(idx - 1) * 17 + 6] = dely;
        total[(idx - 1) * 17 + 7] = delz;
        total[(idx - 1) * 17 + 8] = theta_x;
        total[(idx - 1) * 17 + 9] = theta_y;
        total[(idx - 1) * 17 + 10] = theta_z;
        total[(idx - 1) * 17 + 11] = children_num;
        total[(idx - 1) * 17 + 12] = brother_num;
        total[(idx - 1) * 17 + 13] = volume;
        total[(idx - 1) * 17 + 14] = x_re;
        total[(idx - 1) * 17 + 15] = y_re;
        total[(idx - 1) * 17 + 16] = z_re;
        node_idx[idx - 1] = idx - 1;

    }
    """)

    process_iteration_kernel = mod.get_function("process_iteration_kernel")

    skel = skel.astype(np.float32)
    mask = mask.astype(np.float32)
    spacing = spacing.astype(np.float32)
    generation = generation.astype(np.int32)
    parent_map = parent_map.astype(np.int32)
    children_map = children_map.astype(np.int32)

    max_label = int(skel.max())
    x_dim, y_dim, z_dim = skel.shape

    node_idx = np.ones(max_label, dtype=np.int32) * -1
    total = np.zeros((max_label, 17), dtype=np.float32)
    skl1 = skl_p = np.zeros_like(skel, dtype=np.int32)

    skl_left_right = lr.astype(np.int32)
    skl_left_right = np.zeros_like(skl_left_right, dtype=np.int32)

    # relational position
    left = (mask == 2).astype(np.int32)
    right = (mask == 1).astype(np.int32)

    bounding_left = np.array(find_bb_3D(left)).astype(np.float32)
    bounding_right = np.array(find_bb_3D(right)).astype(np.float32)

    # Allocate device memory
    skel_gpu = drv.mem_alloc(skel.nbytes)
    generation_gpu = drv.mem_alloc(generation.nbytes)
    spacing_gpu = drv.mem_alloc(spacing.nbytes)
    parent_map_gpu = drv.mem_alloc(parent_map.nbytes)
    children_map_gpu = drv.mem_alloc(children_map.nbytes)
    node_idx_gpu = drv.mem_alloc(node_idx.nbytes)
    total_gpu = drv.mem_alloc(total.nbytes)
    bounding_left_gpu = drv.mem_alloc(bounding_left.nbytes)
    bounding_right_gpu = drv.mem_alloc(bounding_right.nbytes)
    skl1_gpu = drv.mem_alloc(skl1.nbytes)
    left_gpu = drv.mem_alloc(left.nbytes)
    right_gpu = drv.mem_alloc(right.nbytes)
    skl_p_gpu = drv.mem_alloc(skl_p.nbytes)
    mask_gpu = drv.mem_alloc(mask.nbytes)
    skl_left_right_gpu = drv.mem_alloc(skl_left_right.nbytes)

    # Copy data to device
    drv.memcpy_htod(skel_gpu, skel)
    drv.memcpy_htod(generation_gpu, generation)
    drv.memcpy_htod(spacing_gpu, spacing)
    drv.memcpy_htod(parent_map_gpu, parent_map)
    drv.memcpy_htod(children_map_gpu, children_map)
    drv.memcpy_htod(node_idx_gpu, node_idx)
    drv.memcpy_htod(total_gpu, total)
    drv.memcpy_htod(bounding_left_gpu, bounding_left)
    drv.memcpy_htod(bounding_right_gpu, bounding_right)
    drv.memcpy_htod(skl1_gpu, skl1)
    drv.memcpy_htod(left_gpu, left)
    drv.memcpy_htod(right_gpu, right)
    drv.memcpy_htod(skl_p_gpu, skl_p)
    drv.memcpy_htod(mask_gpu, mask)
    drv.memcpy_htod(skl_left_right_gpu, skl_left_right)

    # Launch kernel
    block_size = 256
    grid_size = (max_label + block_size - 1) // block_size
    process_iteration_kernel(
        skel_gpu, generation_gpu, np.float32(
            xt), np.float32(yt), np.float32(zt), spacing_gpu,
        parent_map_gpu, children_map_gpu, node_idx_gpu, total_gpu,
        np.int32(max_label), np.int32(x_dim), np.int32(y_dim), np.int32(z_dim),
        bounding_left_gpu, bounding_right_gpu,
        skl1_gpu, left_gpu, right_gpu, skl_p_gpu, mask_gpu, skl_left_right_gpu,
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    drv.Context.synchronize()  # 确保所有的printf都已输出

    # Copy results back to host
    drv.memcpy_dtoh(node_idx, node_idx_gpu)
    drv.memcpy_dtoh(total, total_gpu)
    drv.memcpy_dtoh(skl1, skl1_gpu)
    drv.memcpy_dtoh(left, left_gpu)
    drv.memcpy_dtoh(right, right_gpu)
    drv.memcpy_dtoh(skl_left_right, skl_left_right_gpu)
    drv.memcpy_dtoh(mask, mask_gpu)

    # Free device memory
    skel_gpu.free()
    generation_gpu.free()
    spacing_gpu.free()
    parent_map_gpu.free()
    children_map_gpu.free()
    node_idx_gpu.free()
    total_gpu.free()
    bounding_left_gpu.free()
    bounding_right_gpu.free()
    skl1_gpu.free()
    left_gpu.free()
    right_gpu.free()
    skl_p_gpu.free()
    mask_gpu.free()
    skl_left_right_gpu.free()
    return total, node_idx, skl_left_right


def find_bb_3D(segmentation: np.ndarray):
    """ Get 3D Bounding box given segmentation mask. """

    if len(segmentation.shape) != 3:
        print("The dimension of input is not 3!")
    pos = np.where(segmentation)

    return pos[0].min(), pos[0].max() + 1, pos[1].min(), pos[1].max() + 1, pos[2].min(), pos[2].max() + 1


def rank(pos):

    new_pos = np.zeros_like(pos)
    for j in range(3):
        tmp = pos[:, j]
        # print(j,tmp)
        tmp_l = tmp[tmp < 0] * (-1)
        tmp_r = tmp[tmp > 0]

        pool_l = np.linspace(0.005, 1, tmp_l.shape[0])
        pool_r = np.linspace(0.005, 1, tmp_r.shape[0])

        rank_l = np.argsort(tmp_l)
        rank_r = np.argsort(tmp_r)

        new_tmp_l = np.zeros_like(tmp_l)
        new_tmp_r = np.zeros_like(tmp_r)
        new_tmp_l[rank_l] = pool_l
        new_tmp_r[rank_r] = pool_r

        new_pos[:, j][tmp < 0] = -new_tmp_l
        new_pos[:, j][tmp > 0] = new_tmp_r
        new_pos[:, j][tmp == 0] = 0
    return new_pos


def normalize(feat):

    max = np.max(feat)
    min = np.min(feat)
    if min >= 0:
        feat = feat / max
    if max < 0:
        feat = -feat / min
    if min < 0 and max > 0:
        delta = feat > 0
        feat = delta * feat / max + (1 - delta) * feat / (-min)
    return feat


def normalize_space(feat):  # feat为N*3

    max = np.max(np.abs(feat))
    feat = feat / max
    return feat


def parent_refine(parent_map, node_idx, parse2node):

    parent_map_new = np.zeros((node_idx.shape[0], node_idx.shape[0]))
    children_map_new = np.zeros((node_idx.shape[0], node_idx.shape[0]))
    for i in range(node_idx.shape[0]):
        parents = np.where(parent_map[node_idx[i], :])[0]
        flag_parent = False
        for parent in parents:
            if parse2node[parent] != -1:
                flag_parent = True
                # print(parse2node[parent])
                parent_map_new[i, parse2node[parent]] = 1
                children_map_new[parse2node[parent], i] = 1
        grands = parents
        while not flag_parent:
            parents = grands
            for parent in parents:
                grands = np.where(parent_map[parent])[0]
                for grand in grands:
                    if parse2node[grand] != -1:
                        flag_parent = True
                        parent_map_new[i, parse2node[grand]] = 1
                        children_map_new[parse2node[grand], i] = 1
    return parent_map_new, children_map_new


def get_edge(parent_map, child_map):

    edge = []
    edge_feature = []
    parent_sum = np.sum(parent_map, axis=1)
    parent_idx = np.where(parent_sum > 1)[0]  # 不只一个父节点的节点
    for i in range(len(parent_idx)):
        parent = np.where(parent_map[parent_idx[i], :])[0]  # 该节点的父节点们
        select = parent[random.randint(0, parent.shape[0] - 1)]
        for parent_i in parent:
            if parent_i != select:
                parent_map[parent_idx[i], parent_i] = 0
                child_map[parent_i, parent_idx[i]] = 0

    for i in range(child_map.shape[0]):
        for j in range(child_map.shape[1]):
            if child_map[i, j] == 1 and i != j:
                edge.append([i, j])
                edge_feature.append(1)
                edge.append([j, i])
                edge_feature.append(-1)
    edge = np.array(edge)
    # print(edge.shape)
    edge = edge.transpose((1, 0))
    edge_feature = np.array(edge_feature)
    return edge, edge_feature


def leftright_pycuda(skel, mask):

    mod = SourceModule("""
    __global__ void left_right_kernel(
        float *skel,  int max_label, int x_dim, int y_dim, int z_dim, float *mask, int* skl_left_right) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= max_label) return;
        //if (i != 1) return;
        int idx = i+1;

        for (int z = 0; z < z_dim; z++) {
            for (int y = 0; y < y_dim; y++) {
                for (int x = 0; x < x_dim; x++) {
                    int k = z + y * z_dim + x * z_dim * y_dim;
                    if(skel[k] == idx){
                        if (mask[k]==2) {
                            atomicExch(&skl_left_right[i], 2);
                        } else if (mask[k]==1) {
                            atomicExch(&skl_left_right[i], 1);
                        }
                    }
                }
            }
        }
    }
    """)

    process_iteration_kernel = mod.get_function("left_right_kernel")

    skel = skel.astype(np.float32)
    mask = mask.astype(np.float32)

    max_label = int(skel.max())
    x_dim, y_dim, z_dim = skel.shape

    skl_left_right = np.zeros(max_label, dtype=np.int32)

    skel_gpu = drv.mem_alloc(skel.nbytes)
    mask_gpu = drv.mem_alloc(mask.nbytes)
    skl_left_right_gpu = drv.mem_alloc(skl_left_right.nbytes)

    # Copy data to device
    drv.memcpy_htod(skel_gpu, skel)
    drv.memcpy_htod(mask_gpu, mask)
    drv.memcpy_htod(skl_left_right_gpu, skl_left_right)

    # Launch kernel
    block_size = 256
    grid_size = (max_label + block_size - 1) // block_size
    process_iteration_kernel(
        skel_gpu, np.int32(max_label), np.int32(
            x_dim), np.int32(y_dim), np.int32(z_dim),
        mask_gpu, skl_left_right_gpu,
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    drv.Context.synchronize()  # 确保所有的printf都已输出

    # Copy results back to host

    drv.memcpy_dtoh(skl_left_right, skl_left_right_gpu)
    drv.memcpy_dtoh(mask, mask_gpu)

    # Free device memory
    skel_gpu.free()
    mask_gpu.free()
    skl_left_right_gpu.free()
    return skl_left_right


def feature_extraction_cuda(
    skel_parse: np.ndarray,
    spacing: List,
    lung_mask: np.ndarray,
    parent_map: np.ndarray,
    children_map: np.ndarray,
    generation: np.ndarray,
    trachea_index: int,
):
    """ PyCUDA accelerated feature extraction on airway skeleton tree. """

    trachea_label: np.ndarray = (skel_parse == trachea_index).astype(np.uint8)

    xlt, xrt, ylt, yrt, zlt, zrt = find_bb_3D(trachea_label)
    xt = (xlt + xrt) / 2
    yt = (ylt + yrt) / 2
    zt = (zlt + zrt) / 2

    lr = np.zeros(int(skel_parse.max()), dtype=np.int32)

    total, node_idx, lr = process_iteration_pycuda(
        skel_parse, generation, xt, yt, zt,
        np.array(spacing), parent_map, children_map, lung_mask, lr)

    idx_filter = np.where(node_idx != -1)[0]
    total = total[idx_filter, :]
    node_idx = node_idx[idx_filter]
    pos_rank = rank(total[:, -3::])
    data_final = np.concatenate((total, pos_rank), axis=1)
    data_raw = data_final.copy()

    # normalize
    for j in range(data_final.shape[1]):
        if j == 0 or 1 <= j <= 3 or 5 <= j <= 7 or j == 11 or j == 12 \
                or j == 14 or j == 15 or j == 16:
            continue
        if j == 8 or j == 9 or j == 10:  # angle
            data_final[:, j] = data_final[:, j] / 180
        else:
            data_final[:, j] = normalize(data_final[:, j])
    data_final[:, 1:4] = normalize_space(data_final[:, 1:4])
    data_final[:, 5:8] = normalize_space(data_final[:, 5:8])

    parse2node = np.ones(1000).astype(int) * (-1)
    for i in range(node_idx.shape[0]):
        parse2node[node_idx[i]] = i

    parent_map_new, children_map_new = parent_refine(
        parent_map, node_idx, parse2node)
    edge, edge_feature = get_edge(parent_map_new, children_map_new)

    return data_final, edge, edge_feature, node_idx, data_raw
