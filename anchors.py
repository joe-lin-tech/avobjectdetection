import torch


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
            anchor_range[0], anchor_range[3], self.feature_map_size[0] - 1)
        centers_y = torch.linspace(
            anchor_range[1], anchor_range[4], self.feature_map_size[1] - 1)
        rotations = torch.tensor(self.anchor_rotations)
        centers_x, centers_y, rotations = torch.meshgrid(
            centers_x, centers_y, rotations)
        tiled_centers_x = torch.tile(centers_x, (self.batch_size, 1, 1, 1, 1))
        tiled_centers_y = torch.tile(centers_y, (self.batch_size, 1, 1, 1, 1))
        tiled_centers_z = torch.ones_like(
            tiled_centers_x) * ((anchor_range[5] - anchor_range[2]) / 2)
        tiled_widths = torch.ones_like(tiled_centers_x) * anchor_size[0]
        tiled_lengths = torch.ones_like(tiled_centers_x) * anchor_size[1]
        tiled_heights = torch.ones_like(tiled_centers_x) * anchor_size[2]
        tiled_rotations = torch.tile(rotations, (self.batch_size, 1, 1, 1, 1))
        anchors = torch.stack((tiled_centers_x, tiled_centers_y, tiled_centers_z,
                               tiled_widths, tiled_lengths, tiled_heights, tiled_rotations), dim=-1)
        return anchors

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
        return torch.cat(multiscale_anchors, dim=2)

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

        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        # boxes1_area, boxes2_area: (M, )

        return boxes_area / (boxes1_area[:, None] + boxes2_area[None, :] - boxes_area)

    def boxes_to_bev(self, boxes):
        """
        Convert boxes to bird's eye view

        Parameter boxes: a tensor containing the boxes
        Return: a tensor containing the boxes in bird's eye view
        """
        # TODO understand this projection process
        
        return

    def match_anchors(self, batched_gt_boxes, batched_gt_labels, thresholds, num_classes):
        """
        Match anchors to ground truth objects

        Parameter batched_gt_boxes: a tensor containing the ground truth boxes
        Parameter batched_gt_labels: a tensor containing the ground truth labels
        Parameter thresholds: a list of tuples containing the thresholds for matching anchors to ground truth objects
        Parameter num_classes: number of classes
        Return: a tensor containing the anchors matched to ground truth objects
        """
        for gt_boxes, gt_labels, multiscale_anchors in zip(batched_gt_boxes, batched_gt_labels, self.batched_multiscale_anchors):
            # TODO implement matching of anchors to ground truth objects
            for i, threshold in enumerate(thresholds):
                pos_iou, neg_iou = threshold['pos_iou_thres'], threshold['neg_iou_thres']
                anchors = multiscale_anchors[:, :, i, :, :].reshape(-1, 7)
                bev_anchors = self.boxes_to_bev(anchors)
        return
