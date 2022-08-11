from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from anchors import AnchorGenerator
from tqdm import tqdm
import random
import sys


class Voxelization(nn.Module):
    """
    A class representing the voxelization layer

    Attribute voxel_size: a list containing the voxel size
    Attribute point_cloud_range: a list containing the point cloud range
    Attribute max_num_points: maximum number of points per voxel
    Attribute max_voxels: maximum number of voxels
    """

    def __init__(self, voxel_size, pointcloud_range, max_num_points, max_voxels):
        """
        Initializing Voxelization object

        Parameter voxel_size: a list containing the voxel size
        Parameter pointcloud_range: a list containing the point cloud range
        Parameter max_num_points: maximum number of points per voxel
        Parameter max_voxels: maximum number of voxels
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pointcloud_range = pointcloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

    def forward(self, pointcloud):
        """
        Voxelization layer

        Parameter pointcloud: point cloud data for a single frame
        Return: point pillars, coordinates, number of points per pillar
        """
        pillars = {}
        for point in tqdm(pointcloud, desc='Voxelization'):
            if self.pointcloud_range[0] <= point[0] <= self.pointcloud_range[3] \
                and self.pointcloud_range[1] <= point[1] <= self.pointcloud_range[4] \
                    and self.pointcloud_range[2] <= point[2] <= self.pointcloud_range[5]:
                pillar_coord = (point[0] // self.voxel_size[0], point[1] //
                                self.voxel_size[1], point[2] // self.voxel_size[2])
                if pillar_coord not in pillars:
                    pillars[pillar_coord] = point[None, :]
                else:
                    pillars[pillar_coord] = torch.cat(
                        (pillars[pillar_coord], point[None, :]), 0)
        if len(pillars) > self.max_voxels:
            randomized_keys = random.sample(
                list(pillars.keys()), self.max_voxels)
            pillars = dict(
                zip(randomized_keys, [pillars[key] for key in randomized_keys]))
        pillar_coords = torch.tensor(list(pillars.keys())).long()
        num_points = torch.tensor(
            [max(pillars[pillar_coord].shape[0], self.max_num_points) for pillar_coord in pillars])
        pillars = torch.stack([F.pad(pillar, (0, 0, 0, self.max_num_points - len(pillar)), 'constant', 0) if len(
            pillar) <= self.max_num_points else random.sample(pillar, self.max_num_points) for pillar in pillars.values()])
        return pillars, pillar_coords, num_points


class PillarLayer(nn.Module):
    """
    A class representing the initial pillaring layer of the PointPillars model

    Attribute voxel_size: size of voxels in meters
    Attribute pointcloud_range: min-max range values of each dimension
    Attribute max_num_points: maximum number of points in a pointcloud voxel
    Attribute num_voxels: amount of voxels in grid for each dimension
    """

    def __init__(self, voxel_size, pointcloud_range, max_num_points, max_voxels):
        """
        Initializing PillarLayer object

        Parameter voxel_size: size of voxels in meters
        Parameter pointcloud_range: min-max range values of each dimension
        Parameter max_num_points: maximum number of points in a pointcloud voxel
        Parameter max_voxels: maximum number of voxels in a pointcloud
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pointcloud_range = pointcloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.voxelize = Voxelization(
            voxel_size, pointcloud_range, max_num_points, max_voxels)

    def forward(self, batched_pc):
        """
        Forward pass through the PillarLayer

        Parameter batched_pc: batched pointcloud data
        Return: pillars, their respective coordinates, and the number of points in each pillar
        """
        batched_pillars, batched_pillar_coords, batched_num_points = [], [], []
        for i, pc in enumerate(batched_pc):
            pillars, pillar_coords, num_points = self.voxelize(pc)
            # pillars: (N, max_num_points, c)
            # pillar_coords: (N, 3)
            # num_points: (N, 1)
            batched_pillars.append(pillars)
            batched_pillar_coords.append(F.pad(pillar_coords, (1, 0), value=i))
            batched_num_points.append(num_points)
        batched_pillars = torch.cat(batched_pillars, dim=0)
        # batched_pillars: (B * N, max_num_points, c)
        batched_pillar_coords = torch.cat(batched_pillar_coords, dim=0)
        # batched_pillar_coords: (B * N, 4)
        batched_num_points = torch.cat(batched_num_points, dim=0)
        # batched_num_points: (B * N, 1)
        return batched_pillars, batched_pillar_coords, batched_num_points


class PillarEncoder(nn.Module):
    """
    A class representing the pillar encoder of the PointPillars model

    Attribute voxel_size: size of voxels in meters
    Attribute pointcloud_range: min-max range values of each dimension
    Attribute in_channels: number of channels in the input
    Attribute out_channels: number of channels in the output
    Attribute conv: 1D convolutional layer
    Attribute bn: batch normalization layer
    """

    def __init__(self, voxel_size, pointcloud_range, in_channels, out_channels):
        """
        Initializing PillarEncoder object

        Parameter voxel_size: size of voxels in meters
        Parameter pointcloud_range: min-max range values of each dimension
        Parameter in_channels: number of channels in the input
        Parameter out_channels: number of channels in the output
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pointcloud_range = pointcloud_range
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, batched_pillars, batched_pillar_coords, batched_num_points):
        """
        Forward pass through the PillarEncoder

        Parameter batched_pillars: batched pillar data
        Parameter batched_pillar_coords: batched pillar coordinates
        Parameter batched_num_points: batched number of points in each pillar
        Return: pillar features
        """
        # calculate offsets from pointcloud centers
        offset_pc_center = batched_pillars[:, :, :3] - torch.sum(
            batched_pillars[:, :, :3], dim=1, keepdim=True) / batched_num_points[:, None, None]
        # offset_pc_center: (B * N, max_num_points, 3)

        # calculate offsets from pillar centers
        offset_pillar_center_x = batched_pillars[:, :, :1] - (
            batched_pillar_coords[:, None, 1:2] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.pointcloud_range[0])
        # offset_pillar_center_x: (B * N, max_num_points, 1)
        offset_pillar_center_y = batched_pillars[:, :, 1:2] - (
            batched_pillar_coords[:, None, 2:3] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.pointcloud_range[1])
        # offset_pillar_center_y: (B * N, max_num_points, 1)

        # concatenate pillar augmented features
        augmented_features = torch.cat(
            [batched_pillars, offset_pc_center, offset_pillar_center_x, offset_pillar_center_y], dim=-1)
        # augmented_features: (B * N, max_num_points, c + 5)

        # calculate embedded feature
        augmented_features = augmented_features.permute(0, 2, 1).contiguous()
        # augmented_features: (B * N, c + 5, max_num_points)
        embedded_features = F.relu(self.bn(self.conv(augmented_features)))

        # perform max pooling across the points in each pillar
        pillar_features = torch.max(embedded_features, dim=-1)[0]

        batch_size = int((batched_pillar_coords[-1, 0] + 1).item())
        batched_pillar_features = torch.zeros(batch_size, int((self.pointcloud_range[3] - self.pointcloud_range[0]) / self.voxel_size[0]), int(
            (self.pointcloud_range[4] - self.pointcloud_range[1]) / self.voxel_size[0]), self.out_channels)
        for i in range(batch_size):
            batch_coords = batched_pillar_coords[batched_pillar_coords[:, 0] == i, :]
            batched_pillar_features[i, batch_coords[:, 0],
                                    batch_coords[:, 1]] = pillar_features[batched_pillar_coords[:, 0] == i]
        batched_pillar_features = batched_pillar_features.permute(
            0, 3, 2, 1).contiguous()
        # batched_pillar_features: (B, out_channels, (max_y - min_y) / voxel_size_y, (max_x - min_x) / voxel_size_x)
        return batched_pillar_features


class Backbone(nn.Module):
    """
    A class representing the backbone of the PointPillars model

    Attribute in_channels: number of channels in the input
    Attribute out_channels: number of channels in the output
    Attribute block_1: first block of the backbone
    Attribute block_2: second block of the backbone
    Attribute block_3: third block of the backbone
    Attribute deconv_block_1: first deconvolutional block of the backbone
    Attribute deconv_block_2: second deconvolutional block of the backbone
    Attribute deconv_block_3: third deconvolutional block of the backbone
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializing Backbone object

        Parameter in_channels: number of channels in the input
        Parameter out_channels: number of channels in the output
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_1 = nn.Sequential(OrderedDict([
            ('block1_conv1', nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            ('block1_bn1', nn.BatchNorm2d(64)),
            ('block1_relu1', nn.ReLU(inplace=True)),
            ('block1_conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('block1_bn2', nn.BatchNorm2d(64)),
            ('block1_relu2', nn.ReLU(inplace=True)),
            ('block1_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('block1_bn3', nn.BatchNorm2d(64)),
            ('block1_relu3', nn.ReLU(inplace=True)),
            ('block1_conv4', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('block1_bn4', nn.BatchNorm2d(64)),
            ('block1_relu4', nn.ReLU(inplace=True)),
        ]))
        self.block_2 = nn.Sequential(OrderedDict([
            ('block2_conv1', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            ('block2_bn1', nn.BatchNorm2d(128)),
            ('block2_relu1', nn.ReLU(inplace=True)),
            ('block2_conv2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            ('block2_bn2', nn.BatchNorm2d(128)),
            ('block2_relu2', nn.ReLU(inplace=True)),
            ('block2_conv3', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            ('block2_bn3', nn.BatchNorm2d(128)),
            ('block2_relu3', nn.ReLU(inplace=True)),
            ('block2_conv4', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            ('block2_bn4', nn.BatchNorm2d(128)),
            ('block2_relu4', nn.ReLU(inplace=True)),
            ('block2_conv5', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            ('block2_bn5', nn.BatchNorm2d(128)),
            ('block2_relu5', nn.ReLU(inplace=True)),
            ('block2_conv6', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            ('block2_bn6', nn.BatchNorm2d(128)),
            ('block2_relu6', nn.ReLU(inplace=True)),
        ]))
        self.block_3 = nn.Sequential(OrderedDict([
            ('block3_conv1', nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            ('block3_bn1', nn.BatchNorm2d(256)),
            ('block3_relu1', nn.ReLU(inplace=True)),
            ('block3_conv2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('block3_bn2', nn.BatchNorm2d(256)),
            ('block3_relu2', nn.ReLU(inplace=True)),
            ('block3_conv3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('block3_bn3', nn.BatchNorm2d(256)),
            ('block3_relu3', nn.ReLU(inplace=True)),
            ('block3_conv4', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('block3_bn4', nn.BatchNorm2d(256)),
            ('block3_relu4', nn.ReLU(inplace=True)),
            ('block3_conv5', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('block3_bn5', nn.BatchNorm2d(256)),
            ('block3_relu5', nn.ReLU(inplace=True)),
            ('block3_conv6', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('block3_bn6', nn.BatchNorm2d(256)),
            ('block3_relu6', nn.ReLU(inplace=True)),
        ]))
        self.deconv_block_1 = nn.Sequential(OrderedDict([
            ('deconvblock1_convtranspose', nn.ConvTranspose2d(
                64, 128, kernel_size=1, stride=1)),
            ('deconvblock1_bn', nn.BatchNorm2d(128)),
            ('deconvblock1_relu', nn.ReLU(inplace=True)),
        ]))
        self.deconv_block_2 = nn.Sequential(OrderedDict([
            ('deconvblock2_convtranspose', nn.ConvTranspose2d(
                128, 128, kernel_size=2, stride=2)),
            ('deconvblock2_bn', nn.BatchNorm2d(128)),
            ('deconvblock2_relu', nn.ReLU(inplace=True)),
        ]))
        self.deconv_block_3 = nn.Sequential(OrderedDict([
            ('deconvblock3_convtranspose', nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=4)),
            ('deconvblock3_bn', nn.BatchNorm2d(128)),
            ('deconvblock3_relu', nn.ReLU(inplace=True)),
        ]))

    def forward(self, batched_pillar_features):
        """
        Forward pass through the backbone of PointPillars

        Parameter batched_pillar_features: (B, c, h, w) tensor
        """
        block_1 = self.block_1(batched_pillar_features)
        # block_1: (B, 64, h, w)
        block_2 = self.block_2(block_1)
        # block_2: (B, 128, h/2, w/2)
        block_3 = self.block_3(block_2)
        # block_3: (B, 256, h/4, w/4)
        deconv_block_1 = self.deconv_block_1(block_1)
        # deconv_block_1: (B, 128, h, w)
        deconv_block_2 = self.deconv_block_2(block_2)
        # deconv_block_2: (B, 128, h, w)
        deconv_block_3 = self.deconv_block_3(block_3)
        # deconv_block_3: (B, 128, h, w)
        return torch.cat([deconv_block_1, deconv_block_2, deconv_block_3], dim=1)


class DetectionHead(nn.Module):
    """
    The detection head of PointPillars

    Attribute num_classes: number of classes to predict
    Attribute num_anchors: number of anchors to predict
    Attribute in_channels: number of channels in the input
    Attribute cls_conv: convolutional head for classification
    Attribute reg_conv: convolutional head for regression
    Attribute dir_conv: convolutional head for determining head
    """

    def __init__(self, num_classes, num_anchors, in_channels):
        """
        Initialize the detection head of PointPillars

        Parameter num_classes: number of classes to predict
        Parameter num_anchors: number of anchors to predict
        Parameter in_channels: number of channels in the input
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        self.cls_conv = nn.Conv2d(
            self.in_channels, self.num_anchors * self.num_classes, kernel_size=1)
        self.reg_conv = nn.Conv2d(
            self.in_channels, self.num_anchors * 7, kernel_size=1)
        self.dir_conv = nn.Conv2d(
            self.in_channels, self.num_anchors * 2, kernel_size=1)

    def forward(self, batched_backbone_features):
        """
        Forward pass through the detection head of PointPillars

        Parameter batched_backbone_features: (B, out_channels * 3, h, w) tensor
        """
        cls_head = self.cls_conv(batched_backbone_features)
        # cls_head: (B, num_anchors * num_classes, h, w)
        reg_head = self.reg_conv(batched_backbone_features)
        # reg_head: (B, num_anchors * 7, h, w)
        dir_head = self.dir_conv(batched_backbone_features)
        # dir_head: (B, num_anchors * 2, h, w)
        return cls_head, reg_head, dir_head


class PointPillars(nn.Module):
    """
    A class representing the PointPillars neural network, which consists of various modules

    Parameter voxel_size: (x, y, z) size of each voxel in the voxel grid
    Parameter pointcloud_range: min-max range values of the point cloud
    Parameter max_num_points: maximum number of points in each voxel
    Parameter max_voxels: maximum number of voxels in the pointcloud
    """

    def __init__(self, voxel_size, pointcloud_range, max_num_points, max_voxels):
        """
        Initializes the PointPillars neural network

        Parameter voxel_size: (x, y, z) size of each voxel in the voxel grid
        Parameter pointcloud_range: min-max range values of the point cloud
        Parameter max_num_points: maximum number of points in each voxel
        Parameter max_voxels: maximum number of voxels in the pointcloud
        """
        super().__init__()
        self.pillar_layer = PillarLayer(
            voxel_size, pointcloud_range, max_num_points, max_voxels)
        self.pillar_encoder = PillarEncoder(
            voxel_size, pointcloud_range, in_channels=9, out_channels=64)
        self.backbone = Backbone(in_channels=64, out_channels=128)
        self.detection_head = DetectionHead(
            num_classes=3, num_anchors=6, in_channels=384)
        # TODO remove hard-coded areas
        self.anchor_generator = AnchorGenerator(
            batch_size=2,
            feature_map_size=[248, 216],
            anchor_ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                           [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                           [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            anchor_sizes=[[0.6, 0.8, 1.73],
                          [0.6, 1.76, 1.73],
                          [1.6, 3.9, 1.56]],
            anchor_rotations=[0, 1.57])

        self.thresholds = [{'pos_iou_thres': 0.5, 'neg_iou_thres': 0.35}, {
            'pos_iou_thres': 0.5, 'neg_iou_thres': 0.35}, {'pos_iou_thres': 0.6, 'neg_iou_thres': 0.45}]

    def forward(self, batched_points, batched_gt_boxes, batched_gt_labels):
        """
        Forward pass through the PointPillars neural network

        Parameter batched_points: (B, N, 3) tensor of points
        Parameter batched_gt_boxes: (B, M, 7) tensor of ground truth boxes
        Parameter batched_gt_labels: (B, M) tensor of ground truth labels
        """
        batched_pillars, batched_pillar_coords, batched_num_points = self.pillar_layer(
            batched_points)
        # batched_pillars: (B * N, max_num_points, c)
        # batched_pillar_coords: (B * N, 4)
        # batched_num_points: (B * N, 1)
        batched_pillar_features = self.pillar_encoder(
            batched_pillars, batched_pillar_coords, batched_num_points)
        # batched_pillar_features: (B, out_channels, h, w)
        batched_backbone_features = self.backbone(batched_pillar_features)
        # batched_backbone_features: (B, out_channels * 3, h, w)
        cls_head, reg_head, dir_head = self.detection_head(
            batched_backbone_features)
        # cls_head: (B, num_anchors * num_classes, h, w)
        # reg_head: (B, num_anchors * 7, h, w)
        # dir_head: (B, num_anchors * 2, h, w)
        cls_head = cls_head.permute(0, 2, 3, 1).reshape(-1, 3)
        reg_head = reg_head.permute(0, 2, 3, 1).reshape(-1, 7)
        dir_head = dir_head.permute(0, 2, 3, 1).reshape(-1, 2)
        anchors = self.anchor_generator.batched_multiscale_anchors
        # anchors: (B, h, w, 2, 7)
        anchor_match = self.anchor_generator.match_anchors(
            batched_gt_boxes=batched_gt_boxes, batched_gt_labels=batched_gt_labels, thresholds=self.thresholds, num_classes=3)
        return cls_head, reg_head, dir_head, anchor_match
