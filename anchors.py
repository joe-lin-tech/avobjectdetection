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
        self.multiscale_anchors = self.generate_multiscale_anchors()

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
