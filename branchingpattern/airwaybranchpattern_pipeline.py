import numpy as np
import os
import pandas as pd
import math
import json
from collections import defaultdict


left = np.array([11,3,4,5,6,8,9,10])
seg_3 = [0, 1, 4, 7, 13, 17]
seg_2 = [2, 3, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16]

SEG_NAME = ['LB1+2', 'LB3', 'LB4', 'LB5', 'LB6', 'LB8', 'LB9', 'LB10', 'RB1',
            'RB2', 'RB3', 'RB4', 'RB5', 'RB6', 'RB7', 'RB8', 'RB9', 'RB10']

CLASS2ANNO = json.load(open('class2anno.json'))["subseg"]
CLASS2ANNO = {int(k): v for k, v in CLASS2ANNO.items()}

blocks_seg = {
    "LLB": list(range(0, 4)),
    "LUB": list(range(4, 8)),
    "RUB": list(range(8, 11)),
    "RMB": list(range(11, 13)),
    "RLB": list(range(13, 18))
}


seg2sub = {
    0: [1, 2, 3, 4, 5, 6, 7],
    1: [8, 9, 10, 11, 12, 13, 14],
    2: [15, 16, 17],
    3: [18, 19, 20],
    4: [21, 22, 23, 24, 25, 26, 27],
    5: [28, 29, 30],
    6: [31, 32, 33],
    7: [34, 35, 36, 37, 38, 39, 40],
    8: [41, 42, 43],
    9: [44, 45, 46],
    10: [47, 48, 49],
    11: [50, 51, 52],
    12: [53, 54, 55],
    13: [56, 57, 58, 59, 60, 61, 62],
    14: [63, 64, 65],
    15: [66, 67, 68],
    16: [69, 70, 71],
    17: [72, 73, 74, 75, 76, 77, 78]
}

cotunk1 = [seg2sub[key][0] for key in sorted(seg2sub)]

cotunk2 = {
    5: [2, 3],
    6: [3, 4],
    7: [2, 4],
    12: [9, 10],
    13: [10, 11],
    14: [9, 11],
    25: [22, 23],
    26: [23, 24],
    27: [22, 24],
    38: [35, 36],
    39: [36, 37],
    40: [35, 37],
    60: [57, 58],
    61: [58, 59],
    62: [57, 59],
    76: [73, 74],
    77: [74, 75],
    78: [73, 75]
}

blocks_inter_sub = {
    "LB1+2, LB3": list(range(1, 15)),
    "LB4, LB5": list(range(15, 21)),
    "LB8-10": list(range(28, 41)),
    "RB1-3": list(range(41, 50)),
    "RB4-5": list(range(50, 56)),
    "RB8-10": list(range(66, 79))
}

blocks_inter_seg = {
    "LB1+2, LB3": [0, 1],
    "LB4, LB5": [2, 3],
    "LB8-10": [5, 6, 7],
    "RB1-3": [8, 9, 10],
    "RB4-5": [11, 12],
    "RB8-10": [15, 16, 17]
}

def Class2Anno(classes):
    """
    Convert class labels to an annotation array.

    Parameters:
        classes (np.ndarray): 1D array of class labels (values 0-126).

    Returns:
        np.ndarray: Array of annotations with shape (N, 3) where:
            - Column 0: (1 for left lung, 2 for right lung).
            - Column 1: segment index.
            - Column 2: Subsegment index. 1:a 2:b 3:c 4: a+b 5: b+c 5: a+c
            For invalid class values (class > 126 or -1), the row is set to -1.
    """
    Anno = np.zeros((classes.shape[0],3))
    for i in range(classes.shape[0]):
        class_i = classes[i]
        if 1<= class_i<= 56:
            Anno[i,0] = 1
            Anno[i,1] = left[math.ceil(class_i/7) -1]
            Anno[i,2] = (class_i - 1)%7
        if 57 <= class_i <= 126:
            Anno[i, 0] = 2
            Anno[i, 1] = math.ceil((class_i - 56)/7)
            Anno[i, 2] = (class_i - 1)%7
        if 126 < class_i or class_i == -1:
            Anno[i, :] = -1
    return Anno

def get_mask_des(edge: np.ndarray, node_num: int) -> np.ndarray:
    """
    Create an ancestor mask matrix for the given tree.

    Parameters:
        edge (np.ndarray): A 2 x E array representing directed edges.
        node_num (int): The total number of nodes.

    Returns:
        np.ndarray: An N x N matrix where M[i, j] == 1 if node i is an ancestor of node j.
    """
    # Initialize an adjacency list and mask matrix.
    adj_list = [[] for _ in range(node_num)]
    mask = np.zeros((node_num, node_num), dtype=int)

    # Build the adjacency list from the edge array.
    for i in range(edge.shape[1]):
        u, v = edge[0, i], edge[1, i]
        adj_list[u].append(v)

    def dfs(node: int, ancestor: int):
        """
        Mark the descendant relationship in the mask.

        Parameters:
            node (int): The current node in the DFS traversal.
            ancestor (int): The ancestor node for which we mark all descendants.
        """
        mask[ancestor][node] = 1
        for child in adj_list[node]:
            dfs(child, ancestor)

    # Run DFS from each node to mark descendant relationships.
    for node in range(node_num):
        dfs(node, node)

    return mask


def build_lca_matrix(edge: np.ndarray, generation: np.ndarray) -> np.ndarray:
    """
    Build a Lowest Common Ancestor (LCA) matrix

    Parameters:
        edge (np.ndarray): A 2 x E array representing directed edges, where each edge (u, v) indicates u is the parent of v.
        generation (np.ndarray): An array where generation[i] is the depth of node i.

    Returns:
        np.ndarray: An N x N matrix where the element at [i, j] is the LCA of nodes i and j.
    """
    N = generation.shape[0]

    # Build the parent array where parent[i] is the parent of node i.
    parent = np.full(N, -1, dtype=int)
    for j in range(edge.shape[1]):
        u, v = edge[0, j], edge[1, j]
        parent[v] = u

    def lca(a: int, b: int) -> int:
        """
        Compute the lowest common ancestor of nodes a and b using the generation information.

        Parameters:
            a (int): First node.
            b (int): Second node.

        Returns:
            int: The lowest common ancestor of a and b.
        """
        # Bring both nodes to the same generation.
        while generation[a] > generation[b]:
            a = parent[a]
        while generation[b] > generation[a]:
            b = parent[b]
        # Move up the tree until the nodes meet.
        while a != b:
            a = parent[a]
            b = parent[b]
        return a

    # Construct the LCA matrix.
    lca_matrix = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            lca_matrix[i, j] = lca(i, j)

    return lca_matrix

def subseg2seg(y0):
    """
       Convert subsegment labels to segment labels.

       Parameters:
           y0 (np.ndarray): Array of subsegment labels.

       Returns:
           np.ndarray: Array of corresponding segment labels.
       """

    y = y0.copy()
    for j in range(y.shape[0]):
        if y0[j] == 0:
            y[j] = 18
        if 0 < y0[j] < 127:
            y[j] = (y[j] - 1) // 7

        if 127 <= y0[j] <= 134:
            y[j] = 19

        if y0[j] >= 134:
            y[j] = 20

        if y0[j] == -1:
            y[j] = 20
    return y


def pattern_seg(segment_labels, generation, descendant_mask, lca_matrix):
    """
     Compute intra-segment patterns that satisfy the cotunk condition.

    Parameters:
        segment_labels (np.ndarray): 1D array of shape (N,), representing the segment labels for each node.
        generation (np.ndarray): 1D array of shape (N,), representing the generation (or depth) of each node.
        descendant_mask (np.ndarray): 2D descendant mask matrix.
        lca_matrix (np.ndarray): 2D LCA matrix.

    Returns:
        dict: Dictionary mapping each block (from blocks_seg) to its clusters.
    """
    num_segments = 18  # Total number of segment labels.

    segment_indices = {i: np.where(segment_labels == i)[0] for i in range(num_segments)}
    seg_cotunk = np.zeros((num_segments, num_segments), dtype=np.int16)


    for i in range(num_segments):
        indices_i = segment_indices[i]
        if indices_i.size == 0:
            seg_cotunk[i, :] = -1
            seg_cotunk[:, i] = -1
            continue

        node_i_tunk = indices_i[np.argmin(generation[indices_i])]

        for j in range(num_segments):
            if i == j:
                continue

            indices_j = segment_indices[j]
            if indices_j.size == 0:
                continue

            node_j_tunk = indices_j[np.argmin(generation[indices_j])]
            lca_ij = lca_matrix[node_i_tunk, node_j_tunk]
            labels_under_lca = segment_labels[descendant_mask[lca_ij, :] == 1]

            # The cotunk condition: LCA label is 18 and all descendant labels are in {i, j, 18}.
            if segment_labels[lca_ij] == 18 and np.all(np.isin(labels_under_lca, [i, j, 18])):
                seg_cotunk[i, j] = 1

        combination_seg = {}

        for block_name, indices in blocks_seg.items():
            # If any segment in the block is missing, skip the block.
            if any(np.all(seg_cotunk[i, :] == -1) for i in indices):
                continue
            clusters = find_clusters(seg_cotunk, indices)
            combination_seg[block_name] = clusters

    return combination_seg


def whether_selected(segment_labels, annotations):
    """
    Determine if the segments meet the filtering criteria.
    For seg_2 segments (only 'a' and 'b'):
      - Both annotation 1(a) and 2(b) must exist.
    For other segments (expecting 'a', 'b', and 'c'):
      - Annotations 1(a), 2(b), and 3(c) must exist.
      - At most one composite annotation (4(a+b), 5(b+c), or 6(a+c)) is present.

    Parameters:
        segment_labels (np.ndarray): 1D array of segment labels.
        annotations (np.ndarray): 1D array of annotation values.

    Returns:
        np.ndarray: Array of selection flags (0 or 1) for each of the 18 segments.
    """
    num_segments = 18
    selected = np.zeros((num_segments, ), dtype=np.int16)


    for i in range(num_segments):
        segment_mask = (segment_labels == i)
        annotation_exists = {
            value: np.any(segment_mask & (annotations == value))
            for value in [0, 1, 2, 3, 4, 5, 6]
        }

        if i in seg_2:
            if annotation_exists[1] and annotation_exists[2]:
                selected[i] = 1
        else:
            if (annotation_exists[1] and annotation_exists[2] and annotation_exists[3] and
                    sum(annotation_exists[val] for val in [4, 5, 6]) < 2):
                selected[i] = 1
    return selected


def pattern_sub(segment_labels, annotations):
    """
    Analyze segments based on their annotation values.

    For seg_2 segments (only 'a' and 'b'), no subsegment cotunk is considered.
    For other segments, if annotations 1, 2, and 3 exist (with at most one composite),
    determine the stem number and subsegment cotunk type.

    Parameters:
        segment_labels (np.ndarray): 1D array of segment labels.
        annotations (np.ndarray): 1D array of annotation values.

    Returns:
        dict: Mapping from segment index to a tuple (stem number, cotunk type).
    """
    num_segments = 18
    result = np.zeros((num_segments, 3), dtype=np.int16)


    for i in range(num_segments):
        segment_mask = (segment_labels == i)
        annotation_exists = {
            value: np.any(segment_mask & (annotations == value))
            for value in [0, 1, 2, 3, 4, 5, 6]
        }


        if i in seg_2:
            if annotation_exists[1] and annotation_exists[2]:
                result[i, 0] = 1

                result[i, 1] = 1 if annotation_exists[0] else 2
            # For seg_2, cotunk type remains 0.
        else:

            if (annotation_exists[1] and annotation_exists[2] and annotation_exists[3] and
                    sum(annotation_exists[val] for val in [4, 5, 6]) < 2):
                result[i, 0] = 1
                for val in [4, 5, 6]:
                    if annotation_exists[val]:
                        result[i, 2] = val - 3# Map composite annotation to cotunk type.
                        break
                result[i, 1] = 1 if annotation_exists[0] else (3 if result[i, 2] == 0 else 2)


    pattern_sub_dict = {}
    for seg_idx in range(result.shape[0]):
        if result[seg_idx, 0] == 1:
            pattern_sub_dict[seg_idx] = (int(result[seg_idx, 1]), int(result[seg_idx, 2]))

    return pattern_sub_dict


def pattern_sub_inter(Anno, y_seg, y_sub, generation, mask_des, lca):
    """
       Analyze inter subsegmental patterns.

       Parameters:
           Anno (np.ndarray): Array of subsegment annotations (values 0-7).
           y_seg (np.ndarray): Segment labels for nodes.
           y_sub (np.ndarray): Subsegment labels for nodes.
           generation (np.ndarray): Generation (depth) of each node.
           mask_des (np.ndarray): Descendant mask matrix.
           lca (np.ndarray): LCA matrix.

       Returns:
           dict: Mapping from block (from blocks_inter_sub) to its uniform clusters.
       """


    indices_by_sub = {i: np.where(y_sub == i)[0] for i in range(79)}
    indices_by_seg = {i: np.where(y_seg == i)[0] for i in range(18)}
    sub_cotunk_inter = np.zeros((79, 79))
    seg_selected = whether_selected(y_seg, Anno)

    for i in range(79):
        if i == 0:
            continue

        idx_i = indices_by_sub[i]
        if idx_i.size == 0:
            continue

        sub_cotunk_inter[i, 0] = 1

        x_i_tunk = idx_i[np.argmin(generation[idx_i])]

        for j in range(1, i):
            idx_j = indices_by_sub[j]
            if idx_j.size == 0:
                continue

            if any(i in v and j in v for v in seg2sub.values()):
                if i in cotunk2 and j in cotunk2[i]:
                    sub_cotunk_inter[i, j] = sub_cotunk_inter[j, i] = 1
                if i in cotunk1 or j in cotunk1:
                    sub_cotunk_inter[i, j] = sub_cotunk_inter[j, i] = 1

            x_j_tunk = idx_j[np.argmin(generation[idx_j])]
            lca_ij = lca[x_i_tunk, x_j_tunk]
            label_mask = y_sub[mask_des[lca_ij, :] == 1]

            if y_sub[lca_ij] == 0 and np.all(np.isin(label_mask, [i, j, 0])):
                sub_cotunk_inter[i, j] = 1

        for k in range(0, 18):
            idx_k = indices_by_seg[k]
            if idx_k.size == 0:
                continue

            if i in seg2sub[k]:
                continue

            x_k_tunk = idx_k[np.argmin(generation[idx_k])]

            lca_ik = lca[x_i_tunk, x_k_tunk]
            t = np.where(mask_des[lca_ik, :] == 1)[0]
            if np.all((y_seg[t] == k) | (y_sub[t] == i) | (y_sub[t] == 0)):
                for s in seg2sub[k]:
                    if indices_by_sub[s].size > 0:
                        sub_cotunk_inter[i, s] = sub_cotunk_inter[s, i] = 1


    pattern_sub_inter_dict = {}
    for block_name, indices in blocks_inter_sub.items():
        if any(np.all(seg_selected[i] == 0) for i in blocks_inter_seg[block_name]):
            continue
        indices = [i for i in indices if 1 in sub_cotunk_inter[i, :]]
        clusters = find_clusters(sub_cotunk_inter, indices)
        pattern_sub_inter_dict[block_name] = cluster_uniform_inter(clusters)

    return pattern_sub_inter_dict



def cluster_uniform_inter(cluster):
    """
    Perform uniform clustering on inter subsegment clusters.

    Parameters:
        cluster (tuple): A tuple of tuples representing clusters.

    Returns:
        tuple: A new cluster tuple after uniform clustering.
    """

    unique_segs = []
    flag = True

    for tup in cluster:
        matched_segs = [seg for seg, sub_list in seg2sub.items() if all(sub in sub_list for sub in tup)]
        if len(matched_segs) != 1:
            flag = False
            break
        unique_segs.append(matched_segs[0])

    if flag:
        unique_segs = np.unique(unique_segs)
        new_cluster = tuple((seg2sub[seg][0],) for seg in unique_segs)
        return new_cluster

    new_cluster = [list(t) for t in cluster]

    for seg, sub_list in seg2sub.items():
        tuple_indices = []
        for idx, tup in enumerate(new_cluster):
            if set(tup) & set(sub_list):
                tuple_indices.append(idx)

        if len(tuple_indices) == 1:
            idx = tuple_indices[0]
            new_tup = [label for label in new_cluster[idx] if label not in sub_list]
            new_tup.append(seg2sub[seg][0])
            new_cluster[idx] = new_tup

    new_cluster = tuple(tuple(t) for t in new_cluster)
    return new_cluster

def find_clusters(tensor, indices):
    """
    Identify clusters (combinations) using the union-find algorithm.

    Parameters:
        tensor (np.ndarray): Matrix indicating connections between segments.
        indices (list or np.ndarray): List of indices representing classes in a block.

    Returns:
        tuple: Sorted tuple of clusters (each cluster is a sorted tuple of indices).
    """
    n = len(indices)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for i in range(n):
        for j in range(i + 1, n):
            if tensor[indices[i], indices[j]] == 1:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(indices[i])

    cluster_list = tuple(sorted(tuple(sorted(cluster)) for cluster in clusters.values()))
    return cluster_list



def convert_clusters_to_names(clusters, name_list):
    """
    Convert cluster indices to their corresponding names.

    Parameters:
        clusters (tuple): A tuple of clusters, each a tuple of indices.
        name_list (list or dict): Mapping from index to name.

    Returns:
        tuple: A tuple of clusters with names instead of numeric indices.
    """
    return tuple(tuple(name_list[i] for i in cluster) for cluster in clusters)



def generate(x, edge, y_sub_new):
    """
       Process a single case to compute segmentation patterns and subsegment patterns.

       Parameters:
           x (np.ndarray): Feature array; generation values are assumed in the first column.
           edge (np.ndarray): Edge array representing the airway graph.
           y_sub_new (np.ndarray): Array of new subsegmentation labels.

       Returns:
           tuple: A tuple (combination_seg, pattern_sub_dict, pattern_sub_inter_dict) containing:
               - combination_seg: Intra-segment clusters.
               - pattern_sub_dict: Subsegment branching types.
               - pattern_sub_inter_dict: Inter subsegment clustering.
       """

    generation = x[:, 0]
    mask_des = get_mask_des(edge, x.shape[0])
    lca = build_lca_matrix(edge, generation)

    # Load label mapping from old labels to new labels
    label_mapping_file = 'label_old2new.json'
    with open(label_mapping_file, 'r') as f:
        label_old2new = json.load(f)
    label_new2old = {int(v): int(k) for k, v in label_old2new.items()}

    y_sub_old = np.array([label_new2old[k] for k in y_sub_new])
    y_seg = subseg2seg(y_sub_old)
    Anno = Class2Anno(y_sub_old)[:, 2].astype(np.int16)


    combination_seg = pattern_seg(y_seg, generation, mask_des, lca)
    pattern_sub_dict = pattern_sub(y_seg, Anno)
    pattern_sub_inter_dict = pattern_sub_inter(Anno, y_seg, y_sub_new, generation, mask_des, lca)

    return combination_seg, pattern_sub_dict, pattern_sub_inter_dict

def generate_counts(paths):
    """
    Analyze all patient cases under given directory paths and aggregate pattern counts.

    Parameters:
        paths (str or list of str): Directory or list of directories, where each contains patient subdirectories.

    Returns:
        tuple: Tuple of dictionaries (seg_combination_counts, sub_counts, sub_inter_counts) where:
            - seg_combination_counts: Maps each block from blocks_seg to a defaultdict; keys are cluster combinations.
            - sub_counts: Maps each segment index (0-17) to a defaultdict with pattern counts.
            - sub_inter_counts: Maps each block from blocks_inter_sub to a defaultdict with inter subsegment counts.
    """
    if isinstance(paths, str):
        paths = [paths]

    patient_list = []
    seg_combination_counts = {
        block: defaultdict(lambda: {"count": 0, "patients": []})
        for block in blocks_seg
    }
    sub_counts = {
        seg_class: defaultdict(lambda: {"count": 0, "patients": []})
        for seg_class in range(18)
    }

    sub_inter_counts = {
        block: defaultdict(lambda: {"count": 0, "patients": []})
        for block in blocks_inter_sub
    }


    for base_path in paths:
        patient_dirs = os.listdir(base_path)
        patient_dirs.sort()

        for patient in patient_dirs:
            patient_list.append(patient)
            patient_path = os.path.join(base_path, patient)

            features_path = os.path.join(patient_path, 'airway_feature_cls.npy')
            x = np.load(features_path)

            graph_cls_path = os.path.join(patient_path, 'airway_graph_cls.npy')
            [_, _, y_sub] = np.load(graph_cls_path)

            edge_path = os.path.join(patient_path, "airway_graph.npy")
            edge = np.load(edge_path, allow_pickle=True)

            combination_seg, pattern_sub_dict, pattern_sub_inter_dict = generate(x, edge, y_sub)

            print("Segmental Pattern of case" + patient + ":")
            for block_name, clusters in combination_seg.items():
                seg_combination_counts[block_name][clusters]["count"] += 1
                seg_combination_counts[block_name][clusters]["patients"].append(patient)
                combo_names = convert_clusters_to_names(clusters, SEG_NAME)
                print(block_name, combo_names)

            print("Subsegmental Pattern of case" + patient + ":")
            for seg_class, pattern in pattern_sub_dict.items():
                sub_counts[seg_class][pattern]["count"] += 1
                sub_counts[seg_class][pattern]["patients"].append(patient)
                print(seg_class, pattern)

            print("Inter Subsegmental Pattern of case" + patient + ":")
            for block_name, clusters in pattern_sub_inter_dict.items():
                sub_inter_counts[block_name][clusters]["count"] += 1
                sub_inter_counts[block_name][clusters]["patients"].append(patient)
                combo_names = convert_clusters_to_names(clusters, CLASS2ANNO)
                print(block_name, combo_names)

    return seg_combination_counts, sub_counts, sub_inter_counts



if __name__ == '__main__':


    path_train = ""
    path_val = ""

    # Process all patients and generate pattern statistics.
    counts_seg_dict, counts_sub_dict, counts_sub_inter_dict = generate_counts([path_train, path_val])

    data_seg = []
    data_sub = []
    data_sub_inter = []


    for block, combo_dict in counts_seg_dict.items():
        total = sum(data["count"] for data in combo_dict.values())
        print(f"{block}:", total)
        for combo, data in combo_dict.items():
            ratio = 100 * data["count"] / total
            combo_names = convert_clusters_to_names(combo, SEG_NAME)
            print(f"  {combo_names} -> {ratio:.2f}, patients: {data['patients']}")
            data_seg.append([block,combo_names, data["count"]])



    for block, combo_dict in counts_sub_dict.items():
        total = sum(data["count"] for data in combo_dict.values())
        print(f"{SEG_NAME[block]}:", total)
        for combo, data in sorted(combo_dict.items(), key=lambda item: item[0]):
            ratio = 100 * data["count"] / total
            print(f"  {combo} -> {ratio:.2f}, patients: {data['patients']}")
            data_sub.append([block, combo, data["count"]])



    for block, combo_dict in counts_sub_inter_dict.items():
        total = sum(data["count"] for data in combo_dict.values())
        print(f"{block}:", total)
        for combo, data in combo_dict.items():
            ratio = 100 * data["count"] / total
            count = data["count"]
            combo_names = convert_clusters_to_names(combo, CLASS2ANNO)
            print(f"{combo_names} -> {count}, patients: {data['patients']}")
            data_sub_inter.append([block, combo_names, count])


