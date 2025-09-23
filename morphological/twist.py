# -*- coding: utf-8 -*-
import numpy as np
import math
from sklearn.decomposition import PCA

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


mod = SourceModule("""
#include <math.h>
extern "C"
__global__ void compute_distances(const float* points, int n,
                                  float Sx, float Sy, float Sz,
                                  float Ex, float Ey, float Ez,
                                  float* distances)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    float Px = points[idx * 3];
    float Py = points[idx * 3 + 1];
    float Pz = points[idx * 3 + 2];

    float SE_x = Ex - Sx;
    float SE_y = Ey - Sy;
    float SE_z = Ez - Sz;
    float SE_len2 = SE_x*SE_x + SE_y*SE_y + SE_z*SE_z;

    float PS_x = Px - Sx;
    float PS_y = Py - Sy;
    float PS_z = Pz - Sz;

    float t = 0.0f;
    if(SE_len2 > 1e-8f)
        t = (PS_x*SE_x + PS_y*SE_y + PS_z*SE_z) / SE_len2;

    float dx, dy, dz;
    if(t < 0.0f){
        dx = PS_x; dy = PS_y; dz = PS_z;
    } else if(t > 1.0f){
        dx = Px - Ex; dy = Py - Ey; dz = Pz - Ez;
    } else {
        float proj_x = Sx + t * SE_x;
        float proj_y = Sy + t * SE_y;
        float proj_z = Sz + t * SE_z;
        dx = Px - proj_x; dy = Py - proj_y; dz = Pz - proj_z;
    }
    distances[idx] = sqrtf(dx*dx + dy*dy + dz*dz);
}
""")
compute_distances = mod.get_function("compute_distances")


def process_branch_pca_pycuda(branch_coords, voxel_size=None):
    branch_coords = np.array(branch_coords, dtype=np.float32)
    if voxel_size is not None:
        voxel_size = np.array(voxel_size, dtype=np.float32)
        phys_coords = branch_coords * voxel_size
    else:
        phys_coords = branch_coords.copy()

    n_points = phys_coords.shape[0]
    if n_points < 2:
        raise ValueError("Branch must contain at least 2 points.")
    
    pca = PCA(n_components=1)
    proj = pca.fit_transform(phys_coords)  # shape (n_points,1)
    idx_min = np.argmin(proj)
    idx_max = np.argmax(proj)
    S = phys_coords[idx_min]
    E = phys_coords[idx_max]
    
    points_flat = phys_coords.astype(np.float32).flatten()
    n = np.int32(n_points)
    points_gpu = drv.mem_alloc(points_flat.nbytes)
    distances_gpu = drv.mem_alloc(n_points * np.float32().nbytes)
    drv.memcpy_htod(points_gpu, points_flat)

    Sx, Sy, Sz = np.float32(S[0]), np.float32(S[1]), np.float32(S[2])
    Ex, Ey, Ez = np.float32(E[0]), np.float32(E[1]), np.float32(E[2])

    block_size = 256
    grid_size = int((n_points + block_size - 1) // block_size)

    compute_distances(points_gpu, n,
                      Sx, Sy, Sz,
                      Ex, Ey, Ez,
                      distances_gpu,
                      block=(block_size, 1, 1), grid=(grid_size, 1, 1))

    distances = np.empty(n_points, dtype=np.float32)
    drv.memcpy_dtoh(distances, distances_gpu)

    points_gpu.free()
    distances_gpu.free()
    
    max_idx = np.argmax(distances)
    max_dev = distances[max_idx]
    P = phys_coords[max_idx]

    PS = P - S
    PE = P - E
    norm_PS = np.linalg.norm(PS)
    norm_PE = np.linalg.norm(PE)
    if norm_PS < 1e-8 or norm_PE < 1e-8:
        twist_angle = 0.0
    else:
        cos_val = np.dot(PS, PE) / (norm_PS * norm_PE)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        twist_angle = math.acos(cos_val)

    return S, E, max_dev, P, twist_angle


def point_3d(A):

    A = A.astype(np.float32)
    width, height, depth = A.shape
    total_voxels = width * height * depth

    max_lbl = int(np.max(A))
    if max_lbl == 0:
        return {}
    A_gpu = drv.mem_alloc(A.nbytes)
    drv.memcpy_htod(A_gpu, A)

    counts_host = np.zeros(max_lbl + 1, dtype=np.int32)
    counts_gpu = drv.mem_alloc(counts_host.nbytes)
    drv.memcpy_htod(counts_gpu, counts_host)

    mod = SourceModule("""
            __global__ void count_labels(float* A, int* counts, int max_lbl,
                                       int width, int height, int depth) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total = width * height * depth;
                if (idx >= total) return;

                int k = idx / (width * height);
                int rem = idx % (width * height);
                int j = rem / width;
                int i = rem % width;

                int lbl = (int)A[idx];
                if (lbl > 0 && lbl <= max_lbl) {
                    atomicAdd(&counts[lbl], 1);
                }
            }
        """)
    count_labels = mod.get_function("count_labels")

    block_size = 256
    grid_size = (total_voxels + block_size - 1) // block_size
    count_labels(A_gpu, counts_gpu, np.int32(max_lbl),
                 np.int32(width), np.int32(height), np.int32(depth),
                 block=(block_size, 1, 1), grid=(grid_size, 1, 1))

    drv.memcpy_dtoh(counts_host, counts_gpu)

    offsets = np.zeros(max_lbl + 1, dtype=np.int32)
    current_offset = 0
    for lbl in range(1, max_lbl + 1):
        offsets[lbl] = current_offset
        current_offset += counts_host[lbl]
    total_coords = current_offset

    output_host = np.zeros(total_coords * 3, dtype=np.int32)
    current_counts = np.zeros_like(counts_host)

    offsets_gpu = drv.mem_alloc(offsets.nbytes)
    current_counts_gpu = drv.mem_alloc(current_counts.nbytes)
    output_gpu = drv.mem_alloc(output_host.nbytes)
    drv.memcpy_htod(offsets_gpu, offsets)
    drv.memcpy_htod(current_counts_gpu, current_counts)

    mod = SourceModule("""
            __global__ void fill_coords(float* A, int* offsets, int* current_counts,
                                      int* output, int max_lbl,
                                      int width, int height, int depth) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total = width * height * depth;
                if (idx >= total) return;

                int k = idx / (width * height);
                int rem = idx % (width * height);
                int j = rem / width;
                int i = rem % width;

                int lbl = (int)A[idx];
                if (lbl > 0 && lbl <= max_lbl) {
                    int pos = atomicAdd(&current_counts[lbl], 1);
                    int base = offsets[lbl];
                    output[3*(base + pos)] = i;     
                    output[3*(base + pos)+1] = j;   
                    output[3*(base + pos)+2] = k;  
                }
            }
        """)
    fill_coords = mod.get_function("fill_coords")

    fill_coords(A_gpu, offsets_gpu, current_counts_gpu, output_gpu, np.int32(max_lbl),
                np.int32(width), np.int32(height), np.int32(depth),
                block=(block_size, 1, 1), grid=(grid_size, 1, 1))

    drv.memcpy_dtoh(output_host, output_gpu)

    branch_points = {}
    for lbl in range(1, max_lbl + 1):
        start = offsets[lbl]
        count = counts_host[lbl]
        if count == 0:
            continue
        coords = []
        for i in range(start, start + count):
            idx = 3 * i
            coords.append((output_host[idx],
                           output_host[idx + 1],
                           output_host[idx + 2]))
        branch_points[lbl] = coords

    A_gpu.free()
    counts_gpu.free()
    offsets_gpu.free()
    current_counts_gpu.free()
    output_gpu.free()
    return branch_points


def process_all_branches_with_pca(A, voxel_size=[1.0, 1.0, 1.0]):
    branch_points = point_3d(A)
    N = int(A.max())
    branch_results = np.zeros((N, 2))
    for lbl, pts in branch_points.items():
        pts_arr = np.array(pts, dtype=np.float32)
        S, E, max_dev, P, twist_angle = process_branch_pca_pycuda(
            pts_arr, voxel_size)
        branch_results[lbl - 1] = [max_dev, twist_angle]

    return branch_results
