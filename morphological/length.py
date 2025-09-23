# -*- coding: utf-8 -*-
import numpy as np


def find_bb_3D(mask: np.ndarray):

    if len(mask.shape) != 3:
        raise ValueError("The dimension of input is not 3!")

    pos = np.where(mask)
    return pos[0].min(), pos[0].max() + 1, pos[1].min(), pos[1].max() + 1, \
        pos[2].min(), pos[2].max() + 1


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
    if not nodes:
        return None
    current = nodes[0]
    for node in nodes[1:]:
        current = lca(current, node, parent, generation)
    return current


def get_path(start, end, parent):
    path = []
    node = end
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path


def recursive_find_paths(leaves, parent, generation, labels, lengths, K):
    if len(leaves) == 1:
        path = []
        node = leaves[0]
        while labels[node] == K:
            path.append(node)
            node = parent[node]
        return [sum(lengths[node] for node in path)]

    candidate = compute_lca(leaves, parent, generation)
    if candidate is None:
        return []

    if labels[candidate] == K:
        path_sums = []
        for leaf in leaves:
            path = get_path(candidate, leaf, parent)
            s = sum(lengths[node] for node in path)
            path_sums.append(s)
        return path_sums
    else:
        path_sums = []
        for leaf in leaves:
            path = get_path(candidate, leaf, parent)
            new_candidate = None
            for node in path[1:]: 
                if labels[node] == K:
                    new_candidate = node
                    break
            if new_candidate is None:
                new_candidate = leaf
            path = get_path(new_candidate, leaf, parent)
            s = sum(lengths[node] for node in path)
            path_sums.append(s)

        return path_sums


def analyze_label_K_paths(edge, generation, labels, lengths, K):
    N = len(labels)
    parent, children = get_parent_children_map(edge, N)
    K_leaves = []
    if labels[labels == K].shape[0] == 0:
        return -1
    for i in range(N):
        if labels[i] == K and len(children[i]) == 0:
            K_leaves.append(i)

    if not K_leaves:
        for node in range(N):
            if labels[node] == K:
                has_K_child = False
                for c in children[node]:
                    if labels[c] == K:
                        has_K_child = True
                        break
                if not has_K_child:
                    K_leaves.append(node)

    path_sums = recursive_find_paths(
        K_leaves, parent, generation, labels, lengths, K)
    if not path_sums:
        return None

    avg_path_length = sum(path_sums) / len(path_sums)
    return avg_path_length
