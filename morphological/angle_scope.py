# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import differential_evolution


def get_parent_children_map(edge, N):

    parent = [None] * N
    children = [[] for _ in range(N)]
    for i in range(edge.shape[1]):
        node1 = edge[0, i]
        node2 = edge[1, i]
        parent[node2] = node1
        children[node1].append(node2)

    return parent, children


def lca(a, b, parent, generation):
    while generation[a] > generation[b]:
        a = parent[a]
    while generation[b] > generation[a]:
        b = parent[b]
    while a != b:
        a = parent[a]
        b = parent[b]
    return a


def compute_lca(nodes, parent, generation):
    current = nodes[0]
    for node in nodes[1:]:
        current = lca(current, node, parent, generation)
    return current


def compute_max_open_angle(edge, generation, labels, coords, category):
    N = len(labels)
    if labels[labels == category].shape[0] == 0:
        return -1
    parent, children = get_parent_children_map(edge, N)
    K_leaves = []

    for i in range(N):
        if labels[i] == category and len(children[i]) == 0:
            K_leaves.append(i)

    if not K_leaves:
        for node in range(N):
            if labels[node] == category:
                has_K_child = False
                for c in children[node]:
                    if labels[c] == category:
                        has_K_child = True
                        break
                if not has_K_child:
                    K_leaves.append(node)
    if len(K_leaves) == 1:
        return 0
    else:
        apex = compute_lca(K_leaves, parent, generation)

    apex_coord = coords[apex]
    d_vecs = []
    for leaf in K_leaves:
        vec = coords[leaf] - apex_coord
        norm = np.linalg.norm(vec)
        if norm > 0:
            d_vecs.append(vec / norm)
    if len(d_vecs) == 0:
        return None

    def objective(x):
        theta, phi = x
        u = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        dots = np.dot(d_vecs, u)
        return -np.min(dots)

    bounds = [(0, np.pi), (0, 2 * np.pi)]
    result = differential_evolution(objective, bounds, tol=1e-6)


    optimum = -result.fun

    optimum = max(min(optimum, 1.0), -1.0)

    half_angle = np.arccos(optimum)

    open_angle = 2 * half_angle
    return open_angle/np.pi


def compute_centroid_and_direction(edge, coor, length, labels, K, generation):
    indices = np.where(labels == K)[0]
    if len(indices) == 0:
        return -1, -1, -1

    total_weight = np.sum(length[indices])
    centroid = np.sum(length[indices, None] *
                      coor[indices], axis=0) / total_weight

    N = labels.shape[0]
    parent, children = get_parent_children_map(edge, N)

    lca_idx = compute_lca(indices, parent, generation)

    if len(indices) == 1:
        lca_idx = parent[lca_idx]

    lca_coor = coor[lca_idx]

    vec = centroid - lca_coor
    norm_vec = np.linalg.norm(vec)
    if norm_vec < 1e-8:
        return 0, 0, 0
    vec_unit = vec / norm_vec

    axis_x = np.array([1, 0, 0])
    axis_y = np.array([0, 1, 0])
    axis_z = np.array([0, 0, 1])

    angle_x = np.arccos(np.clip(np.dot(vec_unit, axis_x), -1.0, 1.0))/np.pi
    angle_y = np.arccos(np.clip(np.dot(vec_unit, axis_y), -1.0, 1.0))/np.pi
    angle_z = np.arccos(np.clip(np.dot(vec_unit, axis_z), -1.0, 1.0))/np.pi

    return [angle_x, angle_y, angle_z]
