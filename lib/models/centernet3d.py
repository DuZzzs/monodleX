import os
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from lib.backbones import dla
from lib.backbones.dlaup import DLAUp
from lib.backbones.hourglass import get_large_hourglass_net
from lib.backbones.hourglass import load_pretrian_model
from lib.backbones.pose_resnet import get_pose_net


class CenterNet3D(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', num_class=3, downsample=4, cfg=None):
        """
        CenterNet for monocular 3D object detection.
        :param backbone: the backbone of pipeline, such as dla34.
        :param neck: the necks of detection, such as dla_up.
        :param downsample: the ratio of down sample. [4, 8, 16, 32]
        :param head_conv: the channels of convolution in head. default: 256
        """
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.consider_outside_objs = False
        if cfg['dataset']['type'] == 'KITTI_v2':
            self.consider_outside_objs = True

        self.heads = {'heatmap': num_class, 'offset_2d': 2, 'size_2d' :2, 'depth': 2, 'offset_3d': 2, 'size_3d':3, 'heading': 24}

        self.use_dlaup = True
        if backbone == 'dla34':
            self.backbone = getattr(dla, backbone)(pretrained=True, return_levels=True)
            channels = self.backbone.channels  # channels list for feature maps generated by backbone
            self.first_level = int(np.log2(downsample))
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
            self.neck = DLAUp(channels[self.first_level:], scales_list=scales)   # feature fusion [such as DLAup, FPN]
        elif backbone == 'res18':
            print("using pose_resnet...")
            self.use_dlaup = False
            channels = [16, 32, 64, 128, 256, 512]
            self.first_level = int(np.log2(downsample))
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]

            num_layers = 18
            heads = {}
            self.backbone = get_pose_net(num_layers, heads)
        else:
            raise NotImplementedError

        self.head_conv = 256

        # initialize the head of pipeline, according to heads setting.
        for head in self.heads.keys():
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0, bias=True))

            # initialization
            if 'heatmap' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # edge feature fusion
        self.bn_momentum = 0.1
        self.edge_fusion_kernel_size = 3
        if self.consider_outside_objs:
            self.trunc_heatmap_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size,
                          padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                nn.BatchNorm1d(self.head_conv, momentum=self.bn_momentum),
                nn.Conv1d(self.head_conv, self.heads['heatmap'], kernel_size=1),
            )

            self.trunc_offset3d_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size,
                          padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                nn.BatchNorm1d(self.head_conv, momentum=self.bn_momentum),
                nn.Conv1d(self.head_conv, self.heads['offset_3d'], kernel_size=1),
            )
            self.trunc_offset2d_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size,
                          padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                nn.BatchNorm1d(self.head_conv, momentum=self.bn_momentum),
                nn.Conv1d(self.head_conv, self.heads['offset_2d'], kernel_size=1),
            )
        self.continue_heads = ['heatmap', 'offset_2d', 'offset_3d']
        self.resolution = np.array([1280, 384])  # W * H
        self.output_width = self.resolution[0] // downsample
        self.output_height = self.resolution[1] // downsample

    def forward(self, input, targets=None):
        b, c, h, w = input.shape
        feat = self.backbone(input)
        if self.use_dlaup:
            feat = self.neck(feat[self.first_level:])

        ret = {}
        feature_cls = self.__getattr__('heatmap')[:-1](feat)
        output_cls = self.__getattr__('heatmap')[-1](feature_cls)
        feature_offset2d_reg = self.__getattr__('offset_2d')[:-1](feat)
        output_offset2d_reg = self.__getattr__('offset_2d')[-1](feature_offset2d_reg)
        feature_offset3d_reg = self.__getattr__('offset_3d')[:-1](feat)
        output_offset3d_reg = self.__getattr__('offset_3d')[-1](feature_offset3d_reg)
        if self.consider_outside_objs:
            edge_indices = targets['edge_indices']  # (b,k,2)
            edge_lens = targets['edge_len']  # (b,)
            # normalize
            grid_edge_indices = edge_indices.view(b, -1, 1, 2).float()
            grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (
                        self.output_width - 1) * 2 - 1  # 0~1 -> 0~2 -> -1~1
            grid_edge_indices[..., 1] = grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1

            # apply edge fusion for both offset2d, offset3d, heatmap
            feature_for_fusion = torch.cat((feature_cls, feature_offset2d_reg, feature_offset3d_reg), dim=1)
            edge_features = F.grid_sample(feature_for_fusion, grid_edge_indices, align_corners=True).squeeze(-1)
            # Each edge is different and processed separately, and its dimensions are still not uniform after concat.
            # The above method samples 00 multiple times, which is problematic
            # for k in range(b):
            #     edge_indices

            edge_cls_feature = edge_features[:, :self.head_conv, ...]
            edge_offset2d_feature = edge_features[:, self.head_conv:self.head_conv * 2, ...]
            edge_offset3d_feature = edge_features[:, self.head_conv * 2:, ...]
            edge_cls_output = self.trunc_heatmap_conv(edge_cls_feature)
            edge_offset2d_output = self.trunc_offset2d_conv(edge_offset2d_feature)
            edge_offset3d_output = self.trunc_offset3d_conv(edge_offset3d_feature)
            for k in range(b):
                edge_indice_k = edge_indices[k, :edge_lens[k]]  # remove repeated points
                output_cls[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_cls_output[k, :, :edge_lens[k]]
                output_offset2d_reg[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_offset2d_output[k, :,
                                                                                       :edge_lens[k]]
                output_offset3d_reg[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_offset3d_output[k, :,
                                                                                       :edge_lens[k]]
        ret['heatmap'] = output_cls
        ret['offset_2d'] = output_offset2d_reg
        ret['offset_3d'] = output_offset3d_reg

        for i, head in enumerate(self.heads):
            if head in self.continue_heads:
                continue
            ret[head] = self.__getattr__(head)(feat)

        return ret


    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




if __name__ == '__main__':
    import torch
    net = CenterNet3D(backbone='dla34')
    print(net)

    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)
    print(output.keys())


