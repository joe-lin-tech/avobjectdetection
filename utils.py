import numpy as np
import torch

def collate_fn(batch):
    """
    Collate a batch of data into a batch of tensors

    Parameter batch: batch of data
    """
    batched_pointclouds = []
    batched_gt_boxes = []
    batched_gt_labels = []
    batched_calibs = []
    for frame in batch:
        batched_pointclouds.append(torch.from_numpy(frame['pointcloud']))
        batched_gt_boxes.append(torch.from_numpy(frame['gt_boxes']))
        batched_gt_labels.append(torch.from_numpy(frame['gt_labels']))
        batched_calibs.append(frame['calib'])
    return dict(
        pointclouds=batched_pointclouds,
        gt_boxes=batched_gt_boxes,
        gt_labels=batched_gt_labels,
        calibs=batched_calibs,
    )

def camera_to_lidar(boxes, Tr_velo_to_cam, R0_rect):
    """
    Transform 3D bounding boxes from camera coordinate to lidar coordinate

    Parameter boxes: 3D bounding boxes in camera coordinate
    Parameter tr_velo_to_cam: transformation matrix from lidar to camera coordinate
    Parameter R0_rect: rotation matrix from camera coordinate to rectified camera coordinate
    Return: 3D bounding boxes in lidar coordinate
    """
    l, h, w, r = boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6]
    location_pad = np.hstack((boxes[:, :3], np.ones((boxes.shape[0], 1))))
    transformed = location_pad @ (R0_rect @ Tr_velo_to_cam).T
    return np.hstack((transformed[..., :3], w[:, np.newaxis], l[:, np.newaxis], h[:, np.newaxis], r[:, np.newaxis]))