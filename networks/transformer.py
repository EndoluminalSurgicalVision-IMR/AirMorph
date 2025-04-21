# -*- coding: utf-8 -*-
"""
Created on Tur Sep 12 08:06:16 2024

@author: Li
"""

import torch
from torch import nn
from einops import rearrange
import numpy as np
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention_spd(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, spd, p, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + spd
        attn = self.attend(dots)
        attn = self.dropout(attn)

        mask = torch.ones_like(attn, requires_grad=False)
        a = np.random.binomial(1, 1 - p, size=mask.shape[1])
        while np.sum(a) == 0:
            a = np.random.binomial(1, 1 - p, size=mask.shape[1])
        for i in range(mask.shape[1]):
            if a[i] == 0:
                mask[:, i, :, :] = 0

        attn = attn * mask * mask.shape[1] / np.sum(a)  # normalization

        if attn_mask is not None:
            attn = attn * attn_mask

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_postnorm_spd(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_spd(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, spd, p, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, spd, p, attn_mask) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x


class learnabel_mask(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.pairwise_processor = PairwiseProcessing(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),  # 输出层
            nn.Sigmoid()  # 将输出限制在0-1之间
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, toplogy_mask):
        K = self.mlp(x).repeat(1, 1, x.shape[1])  # 1*N*N
        x = x.squeeze(0)
        xpair1 = x.unsqueeze(0).repeat(x.shape[0], 1, 1)  # N*N*d
        xpair2 = x.unsqueeze(1).repeat(1, x.shape[0], 1)  # N*N*d
        nodepair = torch.cat([xpair1, xpair2], 2)  # N*N*2d
        nodepair = nodepair.unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # 1*2d*N*N
        nodepair = self.pairwise_processor(nodepair)  # Process pairwise features
        nodepair = nodepair.permute(0, 2, 3, 1).contiguous()  # 1*N*N*2
        nodepair = self.softmax(nodepair)[:, :, :, 0]
        nodepair = toplogy_mask * (nodepair + (1 - nodepair) * K) + (1 - toplogy_mask) * nodepair
        return nodepair


class Attention_cross(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2, spd=None):
        q = self.to_q(x1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(x2).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        if spd is None:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + spd
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_postnorm_cross_spd(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_spd(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.cross = Attention_cross(dim, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x, x2, spd=None, p=0, attn_mask=None):
        if spd is not None:
            for attn, ff in self.layers:
                x = attn(x, spd, p, attn_mask) + x
                x = self.norm(x)
                x = ff(x) + x
                x = self.norm(x)
                x = self.cross(x, x2, spd) + x
                x = self.norm(x)
        return x


class Attention_cross_base(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2):
        q = self.to_q(x1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(x2).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_postnorm_cross(nn.Module):  # use 分支2接收att
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_cross_base(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, x2):
        for attn, ff in self.layers:
            x = attn(x, x2) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x


class PairwiseProcessing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pairbn0 = nn.BatchNorm2d(dim * 2)
        self.pairbn1 = nn.BatchNorm2d(dim)
        self.pairbn2 = nn.BatchNorm2d(dim // 2)
        self.pairbn3 = nn.BatchNorm2d(dim // 4)
        self.pairbn4 = nn.BatchNorm2d(dim // 8)

        self.pairconv10 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False)
        self.pairconv11 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.pairconv20 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.pairconv21 = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False)
        self.pairconv3 = nn.Conv2d(dim // 2, dim // 4, kernel_size=1, bias=False)
        self.pairconv4 = nn.Conv2d(dim // 4, dim // 8, kernel_size=1, bias=False)
        self.pairconv5 = nn.Conv2d(dim // 8, 2, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, nodepair):
        nodepair = self.pairbn0(nodepair)  # 1*d*N*C 128
        nodepair = self.relu(self.pairbn1(self.pairconv11(self.relu(self.pairconv10(nodepair)))))  # 1*64*N*C
        nodepair = self.relu(self.pairbn2(self.pairconv21(self.relu(self.pairconv20(nodepair)))))  # 1*32*N*C
        nodepair = self.relu(self.pairbn3(self.pairconv3(nodepair)))  # 1*16*N*C
        nodepair = self.relu(self.pairbn4(self.pairconv4(nodepair)))  # 1*8*N*C
        nodepair = self.pairconv5(nodepair)  # 1*2*N*C
        return nodepair


class Outlier_detect(nn.Module):
    def __init__(self, depth, dim, heads, mlp_dim, dim_head=64, prototype_class=22, dropout=0., ):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.pairwise_processor = PairwiseProcessing(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.LayerNorm(prototype_class * 2),
            nn.Linear(prototype_class * 2, 1),  # 输出层
            nn.Sigmoid()  # 将输出限制在0-1之间
        )
        self.relu = nn.ReLU(inplace=True)
        self.Trans_cross = Transformer_postnorm_cross(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x, logits):
        S = F.softmax(logits, dim=-1)
        S = S ** self.alpha
        H = torch.matmul(S.transpose(-1, -2), x)
        H = self.Trans_cross(H, x)

        x = x.squeeze(0)
        H = H.squeeze(0)
        xpair1 = H.unsqueeze(0).repeat(x.shape[0], 1, 1)  # N*C*d
        xpair2 = x.unsqueeze(1).repeat(1, H.shape[0], 1)  # N*C*d
        nodepair = torch.cat([xpair1, xpair2], 2)  # N*C*2d
        nodepair = nodepair.unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # 1*2d*N*C
        nodepair = self.pairwise_processor(nodepair)  # Process pairwise features
        nodepair = nodepair.permute(0, 2, 3, 1).contiguous()  # 1*N*C*2
        nodepair = nodepair.view(nodepair.shape[0], nodepair.shape[1], -1)  # 1*N*2C
        outlier = self.mlp(nodepair)
        return outlier


class Stage_independent(nn.Module):
    def __init__(self, depth, outlier_depth, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., ):
        super().__init__()
        hierarchy = [depth, depth, depth, depth]
        self.transformer = nn.ModuleList([])
        self.dense_linear = nn.ModuleList(
            [nn.Linear((i + 1) * dim, dim) for i in range(len(hierarchy))]
        )
        for d in hierarchy:
            self.transformer.append(
                Transformer_postnorm_spd(dim, d, heads, dim_head, mlp_dim, dropout)
            )
        self.att_mask = learnabel_mask(dim=dim)
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )

        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes2)
        )
        self.mlp_head3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes3)
        )
        self.outlier = Outlier_detect(outlier_depth, dim, heads, mlp_dim, dim_head=64, dropout=0.,
                                      prototype_class=num_classes2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, spd, p, toplogy_mask):

        x_ = []
        list = []
        list.append(x)
        pred_ = []
        for i in range(len(self.transformer) - 1):
            x = self.dense_linear[i](torch.cat(list, dim=-1))
            if i == 0:
                x = self.transformer[i](x, spd[i], p, None)
                pred = self.mlp_head1(x)
            if i == 1:
                x = self.transformer[i](x, spd[i], p, None)
                pred = self.mlp_head2(x)
                nodepair = self.att_mask(x, toplogy_mask)
                prior = self.softmax(nodepair)
                pred = torch.matmul(prior, pred)
                outlier = self.outlier(x, pred)
                outlier_mask = 1 - (
                        outlier.repeat(1, 1, x.shape[1]) - outlier.transpose(1, 2).repeat(1, x.shape[1], 1)) ** 2

            if i == 2:
                x = self.transformer[i](x, spd[i], p, nodepair * outlier_mask)  # TODO 和下一句可替换
                x = self.transformer[i + 1](x, spd[i], p, None)
                pred = self.mlp_head3(x)
            x_.append(x)
            list.append(x)
            pred_.append(pred)
        return x_[0], x_[1], x_[2], pred_[0], pred_[1], pred_[2], nodepair, outlier


class Stage_guided(nn.Module):
    def __init__(self, input_depth, outlier_depth, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        hierarchy = [input_depth, input_depth, input_depth, input_depth]
        self.transformer = nn.ModuleList([])
        self.dense_linear = nn.ModuleList(
            [nn.Linear((i + 1) * dim, dim) for i in range(len(hierarchy))]
        )
        layer_num = 0
        for d in hierarchy:
            if layer_num >= 2:
                self.transformer.append(
                    Transformer_postnorm_spd(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm_cross_spd(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1
        self.att_mask = learnabel_mask(dim=dim)
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )

        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes2)
        )
        self.mlp_head3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes3)
        )

        self.outlier = Outlier_detect(outlier_depth, dim, heads, mlp_dim, dim_head=64, dropout=0.,
                                      prototype_class=num_classes2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, spd, x2, p, toplogy_mask, outlier):
        x_ = []
        list = []
        # mask_outlier = outlier.repeat(1,1,outlier.shape[1])
        list.append(x)
        pred_ = []
        for i in range(len(self.transformer) - 1):
            x = self.dense_linear[i](torch.cat(list, dim=-1))
            if i == 0:
                x = self.transformer[i](x, x2[i], spd[i], p, None)
                pred = self.mlp_head1(x)
            if i == 1:
                x = self.transformer[i](x, x2[i], spd[i], p, None)
                pred = self.mlp_head2(x)
                nodepair = self.att_mask(x, toplogy_mask)
                prior = self.softmax(nodepair)
                pred = torch.matmul(prior, pred)
                outlier = self.outlier(x, pred)
                outlier_mask = 1 - (outlier.repeat(1, 1, x.shape[1]) - outlier.transpose(1, 2).repeat(1, x.shape[1],
                                                                                                      1)) ** 2
            if i == 2:
                x = self.transformer[i](x, spd[i], p, nodepair * outlier_mask)
                x = self.transformer[i + 1](x, spd[i], p, None)
                pred = self.mlp_head3(x)
            x_.append(x)
            list.append(x)
            pred_.append(pred)

        return x_[0], x_[1], x_[2], pred_[0], pred_[1], pred_[2], nodepair, outlier


class our_net(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., trans_depth=2, outlier_depth=2):
        super().__init__()

        self.accecpt = Stage_guided(trans_depth, outlier_depth, num_classes1, num_classes2, num_classes3, dim, heads,
                                    mlp_dim,
                                    dim_head=dim_head,
                                    dropout=dropout)

        self.give = Stage_independent(trans_depth, outlier_depth, num_classes1, num_classes2, num_classes3, dim, heads,
                                      mlp_dim, dim_head=dim_head,
                                      dropout=dropout)

        self.to_embedding = nn.Sequential(nn.Linear(input_dim, dim))
        self.spatial_pos_encoders = nn.ModuleList([nn.Embedding(30, heads, padding_idx=0) for _ in range(3)])
        self.softmax = nn.Softmax(dim=-1)

    def _get_dict(self, spd):
        """Encodes spatial position and prepares the dict."""
        return [encoder(spd).permute(0, 3, 1, 2) for encoder in self.spatial_pos_encoders]

    def forward(self, x, toplogy_mask, spd, p):
        x = self.to_embedding(x).unsqueeze(0)
        spd = spd.unsqueeze(0)
        dict = self._get_dict(spd)

        # First stage
        feature1_1, feature2_1, feature3_1, x1_1, x2_1, x3_1, node_pair1, outlier_1 = self.give(x, dict, p,
                                                                                                toplogy_mask)

        # Cross-stage input
        x_cross = [feature2_1, feature3_1]
        feature1_2, feature2_2, feature3_2, x1_2, x2_2, x3_2, node_pair2, outlier_2 = self.accecpt(x, dict, x_cross, p,
                                                                                                   toplogy_mask, None)

        # Process and return outputs
        x1_1 = x1_1.squeeze(0)
        x2_1 = x2_1.squeeze(0)
        x3_1 = x3_1.squeeze(0)
        node_pair1 = node_pair1.squeeze(0)
        outlier_1 = outlier_1.squeeze(0)
        outlier_1 = outlier_1.squeeze(-1)

        node_pair2 = node_pair2.squeeze(0)
        x1_2 = x1_2.squeeze(0)
        x2_2 = x2_2.squeeze(0)
        x3_2 = x3_2.squeeze(0)
        outlier_2 = outlier_2.squeeze(0)
        outlier_2 = outlier_2.squeeze(-1)
        return x1_1, x2_1, x3_1, x1_2, x2_2, x3_2, node_pair1, node_pair2, outlier_1, outlier_2
