import torch
from tqdm import tqdm
import numpy as np
import json

class AnchorGenerator():
    """
    A class representing the anchor generator for PointPillars

    Attribute feature_map_sizes: a list of tuples containing the feature map sizes
    Attribute anchor_strides: a list of tuples containing the anchor strides
    Attribute anchor_sizes: a list of tuples containing the anchor sizes
    Attribute anchor_rotations: a list of tuples containing the anchor rotations
    Attribute multiscale_anchors: a tensor containing the multiscale anchors
    """

    def __init__(self, batch_size, feature_map_size, anchor_ranges, anchor_sizes, anchor_rotations):
        """
        Initializing AnchorGenerator object

        Parameter batch_size: batch size of the anchors
        Parameter feature_map_size: a tuple containing the feature map size
        Parameter anchor_ranges: a list of tuples containing the ranges to generate the anchors within
        Parameter anchor_sizes: a list of tuples containing the anchor sizes, which correspond to a different class object
        Parameter anchor_rotations: a list of tuples containing the anchor rotations
        """
        self.batch_size = batch_size
        self.feature_map_size = feature_map_size
        self.anchor_ranges = anchor_ranges
        self.anchor_sizes = anchor_sizes
        self.anchor_rotations = anchor_rotations
        self.batched_multiscale_anchors = self.generate_multiscale_anchors()

    def generate_anchors(self, anchor_range, anchor_size):
        """
        Generate anchors for a specific feature map configuration

        Parameter anchor_range: range to generate the anchors within
        Parameter anchor_size: anchor size corresponding to a class object
        Return: a tensor containing the anchors for the feature map configuration
        """
        # TODO check correct implementation of anchor generation
        centers_x = torch.linspace(
            anchor_range[0], anchor_range[3], self.feature_map_size[0])
        centers_y = torch.linspace(
            anchor_range[1], anchor_range[4], self.feature_map_size[1])
        rotations = torch.tensor(self.anchor_rotations)
        centers_x, centers_y, rotations = torch.meshgrid(
            centers_x, centers_y, rotations)
        tiled_centers_x = torch.tile(centers_x, (self.batch_size, 1, 1, 1, 1))
        tiled_centers_y = torch.tile(centers_y, (self.batch_size, 1, 1, 1, 1))
        tiled_centers_z = torch.ones_like(
            tiled_centers_x) * ((anchor_range[5] + anchor_range[2]) / 2)
        tiled_widths = torch.ones_like(tiled_centers_x) * anchor_size[0]
        tiled_lengths = torch.ones_like(tiled_centers_x) * anchor_size[1]
        tiled_heights = torch.ones_like(tiled_centers_x) * anchor_size[2]
        tiled_rotations = torch.tile(rotations, (self.batch_size, 1, 1, 1, 1))
        anchors = torch.stack((tiled_centers_x, tiled_centers_y, tiled_centers_z,
                               tiled_widths, tiled_lengths, tiled_heights, tiled_rotations), dim=-1)
        return anchors.permute(0, 2, 3, 1, 4, 5).contiguous()

    def generate_multiscale_anchors(self):
        """
        Generate anchors at multiple scales for the feature map

        Parameter batch_size: batch size of the anchors
        Return: a tensor containing the multiscale anchors
        """
        multiscale_anchors = []
        for i in range(len(self.anchor_sizes)):
            # generate anchors for each scale
            multiscale_anchors.append(self.generate_anchors(
                self.anchor_ranges[i], self.anchor_sizes[i]))
        return torch.cat(multiscale_anchors, dim=3)

    def limit_period(self, val, offset=0.5, period=np.pi):
        """
        Limit the value to the period

        Parameter val: value to limit
        Parameter offset: offset to add to the value
        Parameter period: period to limit the value to
        Return: the limited value
        """
        return val - torch.floor(val / period + offset) * period

    def nearest_bev(self, boxes):
        """
        Find nearest anchor-matching box in birds-eye view

        Parameter boxes: a tensor containing bounding boxes
        Return: a tensor containing the nearest anchor-matching box in birds-eye view
        """
        boxes_bev = boxes[:, [0, 1, 3, 4]]
        boxes_θ = self.limit_period(boxes[:, 6], 0.5, np.pi)
        boxes_bev = torch.where(
            torch.abs(boxes_θ[:, None]) <= np.pi / 4, boxes_bev, boxes_bev[:, [0, 1, 3, 2]])

        boxes_xy = boxes_bev[:, :2]
        boxes_wl = boxes_bev[:, 2:]
        boxes_bev = torch.cat(
            [boxes_xy - boxes_wl / 2, boxes_xy + boxes_wl / 2], dim=-1)
        return boxes_bev

    def bev_iou(self, boxes1, boxes2):
        """
        Calculate the intersection over union of boxes in bird's eye view

        Parameter boxes1: a tensor containing the first set of boxes
        Parameter boxes2: a tensor containing the second set of boxes
        Return: a tensor containing the intersection over union of the boxes
        """
        # boxes1: (N, 4), boxes2: (M, 4)
        boxes_x1 = torch.maximum(boxes1[:, 0][:, None], boxes2[:, 0][None, :])
        boxes_y1 = torch.maximum(boxes1[:, 1][:, None], boxes2[:, 1][None, :])
        boxes_x2 = torch.minimum(boxes1[:, 2][:, None], boxes2[:, 2][None, :])
        boxes_y2 = torch.minimum(boxes1[:, 3][:, None], boxes2[:, 3][None, :])
        # boxes_x1, boxes_y1, boxes_x2, boxes_y2: (N, M)

        boxes_width = torch.clamp(boxes_x2 - boxes_x1, min=0)
        boxes_height = torch.clamp(boxes_y2 - boxes_y1, min=0)
        # boxes_width, boxes_height: (N, M)

        boxes_area = boxes_width * boxes_height
        # boxes_area: (N, M)

        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * \
            (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * \
            (boxes2[:, 3] - boxes2[:, 1])
        # boxes1_area, boxes2_area: (N, ), (M, )

        with open("boxes1.json", "w") as f:
            json.dump(boxes1.tolist(), f)
        
        with open("boxes2.json", "w") as f:
            json.dump(boxes2.tolist(), f)

        return boxes_area / (boxes1_area[:, None] + boxes2_area[None, :] - boxes_area + 1e-8)

    def boxes_to_bev_iou(self, boxes, anchors):
        """
        Convert boxes to bird's eye view

        Parameter boxes: a tensor containing the boxes
        Parameter anchors: a tensor containing the anchors
        Return: a tensor containing the ious of the boxes in bird's eye view
        """
        boxes_bev = self.nearest_bev(boxes)
        anchors_bev = self.nearest_bev(anchors)
        iou = self.bev_iou(boxes_bev, anchors_bev)
        # iou: (N, M)
        return iou

    def boxes_to_deltas(self, boxes, anchors):
        """
        Convert boxes to deltas

        Parameter boxes: a tensor containing the boxes
        Parameter anchors: a tensor containing the anchors
        Return: a tensor containing the deltas of the boxes
        """
        diagonal = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

        delta_x = (boxes[:, 0] - anchors[:, 0]) / diagonal
        delta_y = (boxes[:, 1] - anchors[:, 1]) / diagonal
        delta_z = (boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]

        delta_w = torch.log(boxes[:, 3] / anchors[:, 3])
        delta_l = torch.log(boxes[:, 4] / anchors[:, 4])
        delta_h = torch.log(boxes[:, 5] / anchors[:, 5])

        # delta_θ = boxes[:, 6] - anchors[:, 6]
        delta_θ = torch.sin(boxes[:, 6] - anchors[:, 6])

        deltas = torch.stack(
            [delta_x, delta_y, delta_z, delta_w, delta_l, delta_h, delta_θ], dim=1)
        return deltas

    def match_anchors(self, batched_gt_boxes, batched_gt_labels, thresholds, num_classes):
        """
        Match anchors to ground truth objects

        Parameter batched_gt_boxes: a tensor containing the ground truth boxes
        Parameter batched_gt_labels: a tensor containing the ground truth labels
        Parameter thresholds: a list of tuples containing the thresholds for matching anchors to ground truth objects
        Parameter num_classes: number of classes
        Return: a tensor containing the anchors matched to ground truth objects
        """
        batched_multiscale_cls = []
        batched_multiscale_reg = []
        batched_multiscale_dir = []
        batched_multiscale_ambiguity = []
        for gt_boxes, gt_labels, multiscale_anchors in zip(batched_gt_boxes, batched_gt_labels, self.batched_multiscale_anchors):
            multiscale_cls = []
            multiscale_reg = []
            multiscale_dir = []
            multiscale_ambiguity = []
            for i, threshold in enumerate(thresholds):
                pos_iou, neg_iou = threshold['pos_iou_thres'], threshold['neg_iou_thres']
                anchors = multiscale_anchors[:, :, i, :, :].reshape(-1, 7)
                bev_iou = self.boxes_to_bev_iou(gt_boxes, anchors)
    
                max_anchor_iou, max_anchor_iou_idx = torch.max(bev_iou, dim=0)
                max_gt_iou, _ = torch.max(bev_iou, dim=1)

                assigned_gt_boxes = - \
                    torch.ones((anchors.shape[0]), dtype=torch.long)
                assigned_gt_boxes[max_anchor_iou < neg_iou] = 0

                assigned_gt_boxes[max_anchor_iou >=
                                  pos_iou] = max_anchor_iou_idx[max_anchor_iou >= pos_iou]

                for j in range(gt_boxes.shape[0]):
                    if max_gt_iou[j] >= neg_iou:
                        assigned_gt_boxes[max_anchor_iou[j] == max_gt_iou[j]] = j

                pos_flag = assigned_gt_boxes > 0
                neg_flag = assigned_gt_boxes == 0

                # assign anchor labels
                assigned_gt_labels = torch.zeros(
                    (anchors.shape[0]), dtype=torch.long) + num_classes
                assigned_gt_labels[pos_flag] = gt_labels[assigned_gt_boxes[pos_flag]]
                assigned_amibiguity = torch.ones((anchors.shape[0]), dtype=torch.long)
                assigned_amibiguity[pos_flag] = 0
                assigned_amibiguity[neg_flag] = 0

                # assign anchor regressions
                assigned_gt_reg = torch.zeros(
                    (anchors.shape[0], 7))
                assigned_gt_reg[pos_flag] = self.boxes_to_deltas(
                    gt_boxes[assigned_gt_boxes[pos_flag]], anchors[pos_flag])

                # assign anchor direction
                assigned_gt_dir = torch.zeros(
                    (anchors.shape[0]), dtype=torch.long)
                assigned_gt_dir[pos_flag] = torch.clamp(torch.floor(self.limit_period(
                    gt_boxes[assigned_gt_boxes[pos_flag], 6], 0, 2 * np.pi)).long(), 0, 1)

                y, x, s, a, c = multiscale_anchors.shape
                # s: number of scales, a: number of anchors, c: (x, y, z, w, l, h, θ)
                multiscale_cls.append(assigned_gt_labels.reshape(y, x, 1, a))
                multiscale_reg.append(assigned_gt_reg.reshape(y, x, 1, a, -1))
                multiscale_dir.append(assigned_gt_dir.reshape(y, x, 1, a))
                multiscale_ambiguity.append(assigned_amibiguity.reshape(y, x, 1, a))
            batched_multiscale_cls.append(torch.cat(multiscale_cls, dim=-2).reshape(-1))
            batched_multiscale_reg.append(torch.cat(multiscale_reg, dim=-3).reshape(-1, c))
            batched_multiscale_dir.append(torch.cat(multiscale_dir, dim=-2).reshape(-1))
            batched_multiscale_ambiguity.append(torch.cat(multiscale_ambiguity, dim=-2).reshape(-1))
        anchor_match = {
            'batched_cls': torch.stack(batched_multiscale_cls, dim=0),
            'batched_reg': torch.stack(batched_multiscale_reg, dim=0),
            'batched_dir': torch.stack(batched_multiscale_dir, dim=0),
            'batched_ambiguity': torch.stack(batched_multiscale_ambiguity, dim=0)
        }
        return anchor_match
