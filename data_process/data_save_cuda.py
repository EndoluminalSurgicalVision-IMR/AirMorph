
import numpy as np
import pycuda.autoprimaryctx
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import os
import nibabel


def transform_pred_data(parse, node_idx, pred):
    """ Transform predicted anatomical label to image. """

    mod = SourceModule("""
    __global__ void process_labeling(float* parse_pred, float* whether_labeled,
                                     float* label_pred, float* parse, int num_task,
                                     int width, int height, int depth)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_task) return;

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < depth; z++) {
                    int idx = x + width * (y + height * z);
                    int parse_val = parse[idx];
                    if (whether_labeled[i] == 1 && parse_val == i + 1) {
                        parse_pred[idx] = label_pred[i] + 1;
                    }
                }
            }
        }
    }
    """)
    process_labeling = mod.get_function("process_labeling")

    parse = parse.astype(np.float32)

    parse_pred = np.zeros_like(parse, dtype=np.float32)

    whether_labeled = np.zeros(int(parse.max()), dtype=np.float32)
    label_pred = np.zeros(int(parse.max()), dtype=np.float32)

    for j in range(node_idx.shape[0]):
        label_pred[node_idx[j]] = pred[j]
        whether_labeled[node_idx[j]] = 1

    num_task = int(parse.max())
    x_dim, y_dim, z_dim = parse.shape

    parse_gpu = drv.mem_alloc(parse.nbytes)
    label_pred_gpu = drv.mem_alloc(label_pred.nbytes)
    parse_pred_gpu = drv.mem_alloc(parse_pred.nbytes)
    whether_labeled_gpu = drv.mem_alloc(whether_labeled.nbytes)

    drv.memcpy_htod(parse_gpu, parse)
    drv.memcpy_htod(label_pred_gpu, label_pred)
    drv.memcpy_htod(parse_pred_gpu, parse_pred)
    drv.memcpy_htod(whether_labeled_gpu, whether_labeled)

    block_size = 256
    grid_size = (num_task + block_size - 1) // block_size

    process_labeling(parse_pred_gpu, whether_labeled_gpu,
                     label_pred_gpu, parse_gpu, np.int32(num_task),
                     np.int32(x_dim), np.int32(y_dim), np.int32(z_dim),
                     block=(block_size, 1, 1), grid=(grid_size, 1, 1))

    drv.memcpy_dtoh(parse_pred, parse_pred_gpu)

    parse_gpu.free()
    label_pred_gpu.free()
    parse_pred_gpu.free()
    whether_labeled_gpu.free()

    return parse_pred
