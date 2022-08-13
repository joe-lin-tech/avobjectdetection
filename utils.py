import numpy as np
import torch
import numba

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
        batched_gt_boxes.append(torch.from_numpy(frame['gt_boxes']).type(dtype=torch.float32))
        batched_gt_labels.append(torch.from_numpy(frame['gt_labels']))
        batched_calibs.append(frame['calib'])
    return dict(
        batched_pointclouds=batched_pointclouds,
        batched_gt_boxes=batched_gt_boxes,
        batched_gt_labels=batched_gt_labels,
        batched_calibs=batched_calibs,
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

# TODO understand function of the following function
@numba.jit(nopython=True)
def points_in_boxes(points, surfaces):
    N, n = len(points), len(surfaces)
    m = surfaces.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            box_surface_params = surfaces[j]
            for k in range(m):
                a, b, c, d = box_surface_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks

# TODO understand boxes to bev function
def boxes_to_bev(boxes):
    """
    Transform 3D bounding boxes to bird's eye view
    
    Parameter boxes: 3D bounding boxes
    Return: bird's eye view bounding boxes
    """
    centers, dimensions, angle = boxes[:, :2], boxes[:, 3:5], boxes[:, 6]

    bev_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    bev_corners = bev_corners[None, ...] * dimensions[:, None, :]

    sin_rotation, cos_rotation = np.sin(angle), np.cos(angle)

    rotation_matrix = np.array([[cos_rotation, sin_rotation], [-sin_rotation, cos_rotation]])
    rotation_matrix = np.transpose(rotation_matrix, (2, 1, 0))
    bev_corners = bev_corners @ rotation_matrix

    bev_corners += centers[:, None, :]
    return bev_corners.astype(np.float32)