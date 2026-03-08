'''
Date: 2022-02-20 07:55:10
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-02-21 02:48:19
'''

import torch
import torch.nn as nn

class MLPBlockFC(nn.Module):
    def __init__(self, d_points, d_model, p_dropout):
        super(MLPBlockFC, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(d_points, d_model, bias=False),
                                 nn.BatchNorm1d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Dropout(p=p_dropout))

    def forward(self, x):
        return self.mlp(x)


class MLPBlock2D(nn.Module):
    def __init__(self, d_points, d_model):
        super(MLPBlock2D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(d_points, d_model, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.mlp(x)


class MLPBlock1D(nn.Module):
    def __init__(self, d_points, d_model):
        super(MLPBlock1D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.mlp(x)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

class ResMLPBlock1D(nn.Module):
    def __init__(self, d_points, d_model):
        super(ResMLPBlock1D, self).__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(d_model),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Conv1d(d_model, d_points, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(d_points))
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.mlp2(self.mlp1(x)) + x)

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)
