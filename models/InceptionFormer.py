'''
==============================================================
InceptionFormer: Point Cloud Completion
==============================================================
Author: Binhan Luo
Date: 2025-4-17
==============================================================
'''

from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
from utils.model_utils import *

from utils.mm3d_pn2 import furthest_point_sample, gather_points
from utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import grouping_operation

from utils.pointnet_util import index_points, square_distance, query_knn, fps_subsample, get_nearest_index, indexing_neighbor
from utils.ResMLP import ResMLPBlock1D, MLPBlock1D, MLP_Res, MLP_CONV
from torch import einsum

def Point2Patch(num_patches, patch_size, xyz):
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_patches).long()

    centroid_xyz = index_points(xyz, fps_idx)

    dists = square_distance(centroid_xyz, xyz)

    knn_idx = dists.argsort()[:, :, :patch_size]

    return centroid_xyz, fps_idx, knn_idx

class PatchAbstraction(nn.Module):
    def __init__(self, num_patches, patch_size, in_channel, mlp):
        super(PatchAbstraction, self).__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_act = nn.ModuleList()

        self.mlp_res = ResMLPBlock1D(mlp[-1], mlp[-1])

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.mlp_act.append(nn.ReLU(inplace=True))
            last_channel = out_channel

    def forward(self, xyz, feature):
        B, _, C = feature.shape

        centroid_xyz, centroid_idx, knn_idx = Point2Patch(
            self.num_patches, self.patch_size, xyz)

        centroid_feature = index_points(feature, centroid_idx)  # [B, S, C]
        grouped_feature = index_points(feature, knn_idx)  # [B, S, k, C]
        k = grouped_feature.shape[2]

        grouped_norm = grouped_feature - centroid_feature.view(B, self.num_patches, 1, C)  # [B, S, k, C]

        groups = torch.cat([
            centroid_feature.unsqueeze(2).expand(B, self.num_patches, k, C),  # [B, S, k, C]
            grouped_norm  # [B, S, k, C]
        ], dim=-1)  # [B, S, k, 2C]

        groups = groups.permute(0, 3, 2, 1)  # [B, 2C, k, S]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            act = self.mlp_act[i]
            groups = act(bn(conv(groups)))  # [B, D, k, S]

        max_patches = torch.max(groups, 2)[0]  # [B, D, S]
        max_patches = self.mlp_res(max_patches).transpose(1, 2)  # [B, S, D]

        avg_patches = torch.mean(groups, 2).transpose(1, 2)  # [B, S, D]

        return centroid_xyz, max_patches, avg_patches

class GeometricEncoder(nn.Module):
    def __init__(self, mlp_hidden_dim=64, mlp_out_dim=3, n_knn=16):
        super().__init__()
        self.n_knn = n_knn
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_out_dim = mlp_out_dim

        # Input feature dimension calculation:
        # [xyz(3)] + [knn_xyz(k*3)] + [relative_pos(k*3)] + [distances(k)] + [geo_features(3)]
        self.input_dim = 3 + n_knn * 3 + n_knn * 3 + n_knn + 3

        # Feature extraction MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        B, N, _ = x.shape
        device = x.device

        # 1. KNN search [B, N, k]
        knn_idx = query_knn(self.n_knn, x, x, include_self=False).long()

        # 2. Gather neighbor points [B, N, k, 3]
        batch_indices = torch.arange(B, device=device)[:, None, None].expand(-1, N,  self.n_knn)
        # point_indices = torch.arange(N, device=device)[None, :, None].expand(B, -1,  self.n_knn)
        knn_xyz = x[batch_indices, knn_idx, :]

        # 3. Compute local geometric features
        x_expanded = x.unsqueeze(2)  # [B, N, 1, 3]
        relative_pos = knn_xyz - x_expanded  # [B, N, k, 3]
        distances = torch.norm(relative_pos, dim=-1, keepdim=True)  # [B, N, k, 1]

        # Compute covariance matrix [B, N, 3, 3]
        cov_matrix = torch.matmul(
            relative_pos.transpose(-2, -1),  # [B, N, 3, k]
            relative_pos
        ) / (self.n_knn - 1 + 1e-6)

        # Eigenvalue decomposition [B, N, 3]
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        eigenvalues, _ = torch.sort(eigenvalues, dim=-1, descending=True)

        # Compute geometric features [B, N, 3]
        epsilon = torch.tensor(1e-6, device=device)
        lambda1 = eigenvalues[..., 0].unsqueeze(-1)
        lambda2 = eigenvalues[..., 1].unsqueeze(-1)
        lambda3 = eigenvalues[..., 2].unsqueeze(-1)

        linearity = (lambda1 - lambda2) / (lambda1 + epsilon)
        planarity = (lambda2 - lambda3) / (lambda1 + epsilon)
        scattering = lambda3 / (lambda1 + epsilon)
        geo_features = torch.cat([linearity, planarity, scattering], dim=-1)

        # 4. Concatenate all features [B, N, input_dim]
        features = torch.cat([
            x,  # [B, N, 3]
            knn_xyz.reshape(B, N, -1),  # [B, N, k*3]
            relative_pos.reshape(B, N, -1),  # [B, N, k*3]
            distances.reshape(B, N, -1),  # [B, N, k]
            geo_features  # [B, N, 3]
        ], dim=-1)

        # 5. Process through MLP
        outfeature = self.mlp(features)
        return outfeature

class PointSparseAttn(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        # Hyperparameters
        self.n_knn = 16
        self.nhead = nhead

        # Attention components
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)

        # Feed-forward network
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.activation1 = nn.GELU()

        # Normalization and dropout
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        # Projection layers
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

        # Relative position encoding
        self.rel_pos_encoder = nn.Sequential(
            nn.Conv2d(3, d_model_out // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model_out // 4, 1, kernel_size=1)
        )

    def forward(self, src1, src2, pos=None, if_act=False):
        # Input projection
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)
        b, c, n = src1.shape

        # Relative position encoding
        if if_act and pos is not None and self.rel_pos_encoder:
            pos = pos.transpose(1, 2)

            # KNN search
            knn_idx = query_knn(self.n_knn, pos, pos).long()

            # Compute relative positions
            knn_pos = torch.gather(
                pos.unsqueeze(2).expand(-1, -1, n, -1),
                1,
                knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            )
            rel_pos = pos.unsqueeze(2) - knn_pos

            # Compute position bias
            rel_pos = rel_pos.permute(0, 3, 1, 2)
            rel_pos_bias = self.rel_pos_encoder(rel_pos).squeeze(1)

            # Build sparse attention mask
            attn_bias = torch.full((b, n, n), float('-inf'), device=rel_pos.device)
            attn_bias.scatter_(2, knn_idx, rel_pos_bias)
            attn_bias = attn_bias.repeat_interleave(self.nhead, dim=0)
        else:
            attn_bias = None

        # Reshape for multi-head attention
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        # Attention with normalization
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        src12, _ = self.multihead_attn1(
            query=src1,
            key=src2,
            value=src2,
            attn_mask=attn_bias
        )

        # Residual connection and FFN
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        return src1.permute(1, 2, 0)

class PointFractalGenerator(nn.Module):
    def __init__(self, dim, up_factor=2):
        super(PointFractalGenerator, self).__init__()

        self.up_factor = up_factor

        # Feature extraction layers
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 3, layer_dims=[256, dim * 2])

        # Cross-attention layers
        self.sa1 = PointSparseAttn(dim * 2, 512)
        self.sa2 = PointSparseAttn(512, 512)
        self.sa3 = PointSparseAttn(512, dim * up_factor)

        # Upsampling and feature processing
        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim * 2, hidden_dim=dim, out_dim=dim)
        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

        # Convolutional layers
        self.conv_ps = nn.Conv1d(dim * up_factor, dim * up_factor, kernel_size=1)
        self.conv_delta = nn.Conv1d(dim * 2, dim * 1, kernel_size=1)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(dim, 64, kernel_size=1)

        # Input processing layers
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, dim, kernel_size=1)
        self.conv_1 = nn.Conv1d(256, dim, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)

        # Activation
        self.relu = nn.GELU()

    def forward(self, pcd, Gloal_feature, K_prev=None):

        b, _, n_prev = pcd.shape

        # Process input point cloud
        feat_1 = self.conv_x1(self.relu(self.conv_x(pcd)))  # (B, C, N)

        # Process global feature
        feat_g = self.conv_1(self.relu(self.conv_11(Gloal_feature)))  # (B, C, 1)

        # Feature concatenation
        if K_prev is not None:
            H0 = torch.cat([feat_1, K_prev], dim=1)
        else:
            H0 = torch.cat([feat_1, feat_g.repeat(1, 1, feat_1.shape[-1])], dim=1)

        # Cross-attention processing
        H1 = self.sa1(H0, H0, pos=pcd, if_act=True)
        H2 = self.sa2(H1, H1, pos=pcd, if_act=True)
        H3 = self.sa3(H2, H2, pos=pcd, if_act=True)

        # Feature reshaping
        H3 = self.conv_ps(H3).reshape(b, -1, n_prev * self.up_factor)

        # Upsampling and feature fusion
        H_up = feat_1.repeat(1, 1, self.up_factor)
        H_cat = torch.cat([H3, H_up], dim=1)
        H4 = self.conv_delta(H_cat)

        # Final point cloud generation
        pcd_new = self.conv_out(self.relu(self.conv_out1(H4))) + pcd.repeat(1, 1, self.up_factor)

        return pcd_new, H3

class Feature_Extractor(nn.Module):
    def __init__(self, N = 2048):
        super(Feature_Extractor, self).__init__()
        self.nsample = N

        # N：2048； N/2：1024； N/4：512, N/8:256, N/16: 128
        self.Inception_Module1 = PatchAbstraction(int(N / 2), 16, 2 * 3, [64, 64])
        self.Inception_Module2 = PatchAbstraction(int(N / 4), 16, 2 * 64, [128, 128])
        self.Inception_Module3 = PatchAbstraction(int(N / 8), 16, 2 * 128, [256, 256])
        self.Inception_Module4 = PatchAbstraction(int(N / 16), 16, 2 * 256, [512, 512])

        self.LocalGeometryEmbedding1 = GeometricEncoder(64, 64)
        self.LocalGeometryEmbedding2 = GeometricEncoder(64, 64)
        self.LocalGeometryEmbedding3 = GeometricEncoder(64, 64)
        self.LocalGeometryEmbedding4 = GeometricEncoder(64, 64)

        self.MuliHead_SelfAttention1 = PointSparseAttn(64, 64)
        self.MuliHead_SelfAttention2 = PointSparseAttn(128, 128)
        self.MuliHead_SelfAttention3 = PointSparseAttn(256, 256)
        self.MuliHead_SelfAttention4 = PointSparseAttn(512, 512)

        self.Patch_Embedding1 = MLPBlock1D(128 + 64, 64)
        self.Patch_Embedding2 = MLPBlock1D(256 + 64, 128)
        self.Patch_Embedding3 = MLPBlock1D(512 + 64, 256)
        self.Patch_Embedding4 = MLPBlock1D(1024 + 64, 512)

    def forward(self, partial_cloud):
        partial_cloud = partial_cloud.transpose(1, 2).contiguous()

        if partial_cloud.shape[-1] == 3:
            pos = partial_cloud  #  (B, N, 3)
        else:
            pos = partial_cloud[:, :, :3].contiguous()

        l0_xyz = partial_cloud  # l0_xyz:(B, N, 3)
        l0_features = partial_cloud  # l0_features:(B, N, 3)

        l1_xyz, l1_max_features, l1_avg_features = self.Inception_Module1(l0_xyz, l0_features)

        l1_xyz_features = self.LocalGeometryEmbedding1(l1_xyz)

        l1_avg_features = self.MuliHead_SelfAttention1(l1_avg_features.transpose(1, 2), l1_avg_features.transpose(1, 2)).transpose(1, 2)
        l1_features = torch.cat([l1_max_features, l1_avg_features, l1_xyz_features], dim=-1)  # l1_features:(B, N/2, 128)
        l1_features = self.Patch_Embedding1(l1_features.transpose(1, 2)).transpose(1, 2) # l1_features:(B, N/2, 64)

        l2_xyz, l2_max_features, l2_avg_features = self.Inception_Module2(l1_xyz, l1_features)
        # l2_xyz:(B, N/4, 3) l2_max_features:(B, N/4, 128) l2_avg_features:(B, N/4, 128)

        l2_xyz_features = self.LocalGeometryEmbedding2(l2_xyz)

        l2_avg_features = self.MuliHead_SelfAttention2(l2_avg_features.transpose(1, 2), l2_avg_features.transpose(1, 2)).transpose(1, 2)
        l2_features = torch.cat([l2_max_features, l2_avg_features, l2_xyz_features], dim=-1)  # l2_features:(B, N/4, 256)
        l2_features = self.Patch_Embedding2(l2_features.transpose(1, 2)).transpose(1, 2)  # l2_features:(B, N/4, 128)

        l3_xyz, l3_max_features, l3_avg_features = self.Inception_Module3(l2_xyz, l2_features)
        # l3_xyz:(B, N/8, 3) l3_max_features:(B, N/8, 256) l3_avg_features:(B, N/8, 256)

        l3_xyz_features = self.LocalGeometryEmbedding3(l3_xyz)

        l3_avg_features = self.MuliHead_SelfAttention3(l3_avg_features.transpose(1, 2), l3_avg_features.transpose(1, 2)).transpose(1, 2)
        l3_features = torch.cat([l3_max_features, l3_avg_features, l3_xyz_features], dim=-1)  # l3_features:(B, N/8, 512)
        l3_features = self.Patch_Embedding3(l3_features.transpose(1, 2)).transpose(1, 2)  # l3_features:(B, N/8, 256)

        l4_xyz, l4_max_features, l4_avg_features = self.Inception_Module4(l3_xyz, l3_features)
        # l4_xyz:(B, N/16, 3) l4_max_features:(B, N/16, 512) l4_avg_features:(B, N/16, 512)

        l4_xyz_features = self.LocalGeometryEmbedding4(l4_xyz)

        l4_avg_features = self.MuliHead_SelfAttention4(l4_avg_features.transpose(1, 2), l4_avg_features.transpose(1, 2)).transpose(1, 2)
        l4_features = torch.cat([l4_max_features, l4_avg_features, l4_xyz_features], dim=-1)  # l4_features:(B, N/16, 1024)
        l4_features = self.Patch_Embedding4(l4_features.transpose(1, 2)).transpose(1, 2)  # l4_features:(B, N/16, 512)

        Global_feature = torch.max(l4_features, dim=1)[0].unsqueeze(-1)  # Global_feature:(B, N/16, 1)

        return Global_feature, l4_xyz, l4_features

class Seed_Generator(nn.Module):
    def __init__(self, feat_dim=512, seed_dim=128, n_knn=16, factor=2, attn_channel=True):
        super(Seed_Generator, self).__init__()
        self.uptrans = UpTransformer(512, 128, dim=64, n_knn=n_knn, use_upfeat=False, attn_channel=attn_channel,
                                     up_factor=factor, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, Global_feature, xyz, features):

        # 使用 UpTransformer 上采样特征
        x1 = self.uptrans(xyz, features, features, upfeat=None)  # (B, 256, 256)
        # 拼接特征并通过 MLP 处理

        x1 = self.mlp_1(torch.cat([x1, Global_feature.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, Global_feature.repeat((1, 1, x2.size(2)))], 1))  #  (B, 128, 256)
        completion = self.mlp_4(x3)  # (B, 3, 256), x3(B, N/4, 256)
        return completion, x3

class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True,
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):

        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(in_dim=in_channel * 2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        # 位置编码的 MLP
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # 注意力机制的 MLP
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(
                nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor, 1), (up_factor, 1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # 上采样层
        self.upsample1 = nn.Upsample(scale_factor=(up_factor, 1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # 残差连接层
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat):
        """
        前向传播函数。
        参数:
            pos: 点云坐标，形状为 (B, 3, N)。
            key: 输入特征，形状为 (B, in_channel, N)。
            query: 查询特征，形状为 (B, in_channel, N)。
            upfeat: 上采样特征，形状为 (B, in_channel, N)。
        返回:
            上采样后的特征，形状为 (B, out_channel, N * up_factor)。
        """
        # 特征变换
        value = self.mlp_v(torch.cat([key, query], 1))  # (B, dim, N)
        identity = value
        key = self.conv_key(key)  # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        # KNN 搜索
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        # 特征聚合
        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        # 位置编码
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        # 上采样特征编码
        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat)  # (B, dim, N)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn)  # (B, dim, N, k)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        # 注意力机制
        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel)  # (B, dim, N*up_factor, k)
        attention = self.scale(attention)  # Softmax 缩放

        # 特征聚合
        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel  # (B, dim, N, k)
        value = self.upsample1(value)  # (B, dim, N*up_factor, k)
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)

        # 残差连接
        y = self.conv_end(agg)  # (B, out_dim, N*up_factor)
        identity = self.residual_layer(identity)  # (B, out_dim, N)
        identity = self.upsample2(identity)  # (B, out_dim, N*up_factor)

        return y + identity  # 返回上采样后的特征

class Model(nn.Module):
    def __init__(self, args, up_factors=[4, 8]):
        super(Model, self).__init__()

        self.feat_extractor = Feature_Extractor(N=args.num_points)
        self.seed_generator = Seed_Generator(feat_dim=512, seed_dim=128, n_knn=16, factor=2, attn_channel=True)
        # Upsample layers
        up_layers = []
        for i, factor in enumerate(up_factors):
            up_layers.append(PointFractalGenerator(dim=128, up_factor=factor))
        self.up_layers = nn.ModuleList(up_layers)

    def forward_encoder(self, partial_cloud):
        Global_feature, xyz, features = self.feat_extractor(partial_cloud)
        return Global_feature, xyz, features

    def forward_decoder(self, Global_feature, xyz, features, partial_cloud):

        seed, seed_feat = self.seed_generator(Global_feature, xyz, features)
        return seed, seed_feat

    def forward(self, partial_cloud, gt=None, is_training=True):
        """
        Args:
            partial_cloud: (B, N, 3)
        """
        # Encoder
        Global_feature, xyz, features = self.forward_encoder(partial_cloud)

        # Decoder
        seed, seed_feat = self.forward_decoder(Global_feature.contiguous(), xyz.transpose(1, 2).contiguous(),
                                         features.transpose(1, 2).contiguous(), partial_cloud)

        pred_pcds = []
        pred_pcds.append(seed.transpose(1, 2).contiguous())

        pcd = torch.cat([partial_cloud, seed], dim=2)

        pcd = gather_points(pcd, furthest_point_sample(pcd.transpose(1, 2).contiguous(), 512)) # (B, 3, num_p0)
        K_prev = None
        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, Global_feature, K_prev)
            pred_pcds.append(pcd.transpose(1, 2).contiguous())


        if is_training:
            loss3, _ = calc_cd(pred_pcds[2], gt)
            gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(),
                                     furthest_point_sample(gt, pred_pcds[1].shape[1])).transpose(1, 2).contiguous()

            loss2, _ = calc_cd(pred_pcds[1], gt_fine1)
            gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(),
                                      furthest_point_sample(gt_fine1, pred_pcds[0].shape[1])).transpose(1, 2).contiguous()

            loss1, _ = calc_cd(pred_pcds[0], gt_coarse)

            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            return pred_pcds[2], loss3, loss2, loss1, total_train_loss

        else:
            cd_p, cd_t = calc_cd(pred_pcds[-1], gt)
            cd_p_coarse, cd_t_coarse = calc_cd(seed, gt)

            return {
                'out1': seed.transpose(1, 2).contiguous(),
                'out2': pred_pcds[-1],
                'cd_t_coarse': cd_t_coarse,
                'cd_p_coarse': cd_p_coarse,
                'cd_p': cd_p,
                'cd_t': cd_t,
            }


if __name__ == '__main__':
    print(1)

