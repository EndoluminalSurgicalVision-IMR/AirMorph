import os
import numpy as np
import torch
from torch_geometric.data import Data


def get_fath(root, edge):
    N = edge.max() + 1
    fath = np.ones((N, 15), dtype=np.int16) * -1
    fath[np.where(fath[:, 0] == -1), 0] = root

    for j in range(edge.shape[0]):
        fath[edge[1, j], 0] = edge[0, j]

    fath[root, :] = root

    for j in range(1, 15):
        for i in range(N):
            fath[i, j] = fath[fath[i, j-1], j-1]
    if (np.sum(np.where(fath[:, -1] != root)[0])):
        print(root, fath[:, -1])
    return fath


def get_lca(depth, fa, u, v):
    if depth[u] > depth[v]:
        u, v = v, u  
    for j in range(0, 15):
        if depth[fa[v][j]] == depth[u]:
            v = fa[v][j]
            break
    if u == v:
        return u
    k = 0
    for i in range(15):
        k = i
        if fa[u][i] == fa[v][i]:
            break
    return fa[u][k]


def get_lca_d(lca, depth):
    lca_d = np.zeros_like(lca)
    for i in range(lca_d.shape[0]):
        for j in range(lca_d.shape[1]):
            lca_d[i, j] = depth[i] - depth[lca[i, j]]
    return lca_d


def generation_dict(x):

    node_num = x.shape[0]
    dict = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            dict[i][j] = abs(x[i, 0]-x[j, 0])
    return dict


def floyd(edge_index):
    """ Implementation of Floyd algorithm. """

    node_num = np.max(edge_index) + 1
    adj = np.full((node_num, node_num), np.inf)
    for i in range(node_num):
        adj[i, i] = 0
    for idx in range(edge_index.shape[1]):
        adj[edge_index[0][idx]][edge_index[1][idx]] = 1
        adj[edge_index[1][idx]][edge_index[0][idx]] = 1
    a = adj.copy()

    for k in range(node_num):
        for i in range(node_num):
            for j in range(node_num):
                if a[i][j] > a[i][k]+a[k][j]:
                    a[i][j] = a[i][k]+a[k][j]
    return a


def get_trachea_index(x):

    idx = np.argmax(x[:, 13])
    return idx


def dfs(node, ancestor, adj_list, M):

    M[ancestor][node] = 1
    for child in adj_list[node]:
        dfs(child, ancestor, adj_list, M)


def get_mask(edge, node_num):


    adj_list = [[] for _ in range(node_num)]
    M = np.zeros((node_num, node_num), dtype=int)


    for i in range(edge.shape[1]):
        adj_list[edge[0, i]].append(edge[1, i])

    for node in range(node_num):
        dfs(node, node, adj_list, M)
    return M


def multitask_dataset(
    patient: str,
    x: np.ndarray,
    edge: np.ndarray,
    edge_prop: np.ndarray,
) -> Data:
    """ Build torch_geometric Data for GNN """

    edge = edge[:, edge_prop > 0]
    mask_top = get_mask(edge, x.shape[0])
    mask_top = torch.from_numpy(mask_top).long()
    mask_top.requires_grad = False

    trachea_ind = get_trachea_index(x)

    spd = floyd(edge)
    spd = np.where(spd > 29, 29, spd)
    spd = torch.from_numpy(spd).long()

    gen = torch.from_numpy(generation_dict(x)).long()
    x_new = x[:, 0:11]

    x = (torch.from_numpy(x_new)).float()
    data = Data(x=x, trachea=trachea_ind, patient=patient,
                spd=spd, gen=gen, mask_top=mask_top)

    return data
