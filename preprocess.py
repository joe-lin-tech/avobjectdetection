import os
from tqdm import tqdm
import numpy as np
import pickle
import cv2
from utils import camera_to_lidar, points_in_boxes

def process_calib(calib_path):
    """
    Process the calibration file for a frame

    Parameter calib_path: path to the calibration file
    Return: a dictionary containing the calibration information for the frame
    """
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    calib = {}
    lines = [line.strip().split(' ') for line in lines]
    P0 = np.array(lines[0][1:]).astype(np.float32).reshape(3, 4)
    P1 = np.array(lines[1][1:]).astype(np.float32).reshape(3, 4)
    P2 = np.array(lines[2][1:]).astype(np.float32).reshape(3, 4)
    P3 = np.array(lines[3][1:]).astype(np.float32).reshape(3, 4)
    
    R0_rect = np.array(lines[4][1:]).astype(np.float32).reshape(3, 3)
    Tr_velo_to_cam = np.array(lines[5][1:]).astype(np.float32).reshape(3, 4)
    Tr_imu_to_velo = np.array(lines[6][1:]).astype(np.float32).reshape(3, 4)

    calib['P0'] = np.vstack((P0, np.array([0, 0, 0, 1])))
    calib['P1'] = np.vstack((P1, np.array([0, 0, 0, 1])))
    calib['P2'] = np.vstack((P2, np.array([0, 0, 0, 1])))
    calib['P3'] = np.vstack((P3, np.array([0, 0, 0, 1])))

    R0_rect = np.vstack((R0_rect, np.array([0, 0, 0])))
    calib['R0_rect'] = np.hstack((R0_rect, np.array([[0], [0], [0], [1]])))
    calib['Tr_velo_to_cam'] = np.vstack((Tr_velo_to_cam, np.array([0, 0, 0, 1])))
    calib['Tr_imu_to_velo'] = np.vstack((Tr_imu_to_velo, np.array([0, 0, 0, 1])))
    return calib

def process_lidar(velodyne_path, R0_rect, Tr_velo_to_cam, P2, image_shape):
    """
    Process the velodyne point cloud for a frame

    Parameter velodyne_path: path to the velodyne point cloud
    Parameter R0_rect: rotation matrix from rectified camera to velodyne
    Parameter Tr_velo_to_cam: transformation matrix from velodyne to camera
    Parameter P2: projection matrix for the second camera
    Parameter img_shape: shape of the image
    Return: a numpy array containing the processed point cloud
    """
    lidar_pc = np.reshape(np.fromfile(velodyne_path, dtype=np.float32), (-1, 4))

    # TODO figure out the following projection matrix to CRT kitti functionality
    CR = P2[0:3, 0:3]
    CT = P2[0:3, 3]
    R_inv_C_inv = np.linalg.inv(CR)
    R_inv, C_inv = np.linalg.qr(R_inv_C_inv)
    C = np.linalg.inv(C_inv)
    R = np.linalg.inv(R_inv)
    T = C_inv @ CT

    image_box = [0, 0, image_shape[1], image_shape[0]]

    # TODO understand frustum retrieving function
    fku, fkv = C[0, 0], -C[1, 1]
    u0_v0 = C[0:2, 2]
    near_clip, far_clip = 0.001, 100
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    box_corners = np.array([[image_box[0], image_box[1]], [image_box[0], image_box[3]], [image_box[2], image_box[3]], [image_box[2], image_box[1]]], dtype=C.dtype)
    near_box_corners = (box_corners - u0_v0) / np.array([fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0_v0) / np.array([fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)
    frustum = np.concatenate([ret_xy, z_points], axis=1)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    extended_xyz = np.pad(frustum.T[None, ...], ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(R0_rect @ Tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    frustum = xyz[..., :3]
    rect1 = np.stack([frustum[:, 0], frustum[:, 1], frustum[:, 3], frustum[:, 2]], axis=1)
    rect2 = np.stack([frustum[:, 4], frustum[:, 7], frustum[:, 6], frustum[:, 5]], axis=1)
    rect3 = np.stack([frustum[:, 0], frustum[:, 4], frustum[:, 5], frustum[:, 1]], axis=1)
    rect4 = np.stack([frustum[:, 2], frustum[:, 6], frustum[:, 7], frustum[:, 3]], axis=1)
    rect5 = np.stack([frustum[:, 1], frustum[:, 5], frustum[:, 6], frustum[:, 2]], axis=1)
    rect6 = np.stack([frustum[:, 0], frustum[:, 3], frustum[:, 7], frustum[:, 4]], axis=1)
    group_vertices = np.stack([rect1, rect2, rect3, rect4, rect5, rect6], axis=1)

    # TODO understand the plane equation's function
    vectors = group_vertices[:, :, :2] - group_vertices[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1])
    normal_d = np.einsum('ijk,ijk->ij', group_vertices[:, :, 0], normal_vectors)
    frustum_surfaces = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    indices = points_in_boxes(lidar_pc[:, :3], frustum_surfaces)
    reduced_lidar_pc = lidar_pc[indices.reshape([-1])]
    return reduced_lidar_pc

def process_labels(label_path):
    """
    Process the labels file for a frame

    Parameter label_path: path to the labels file
    Return: a dictionary containing the label information for the frame
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = {}
    name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y = [], [], [], [], [], [], [], []
    for line in lines:
        line = line.strip().split(' ')
        name.append(line[0])
        truncated.append(float(line[1]))
        occluded.append(int(line[2]))
        alpha.append(float(line[3]))
        bbox.append([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
        dimensions.append([float(line[10]), float(line[8]), float(line[9])])
        location.append([float(line[11]), float(line[12]), float(line[13])])
        rotation_y.append(float(line[14]))

    labels['name'] = np.array(name)
    labels['truncated'] = np.array(truncated)
    labels['occluded'] = np.array(occluded)
    labels['alpha'] = np.array(alpha)
    labels['bbox'] = np.array(bbox)
    labels['dimensions'] = np.array(dimensions)
    labels['location'] = np.array(location)
    labels['rotation_y'] = np.array(rotation_y)
    return labels

def generate_dataset_info(root, mode):
    """
    Generate a list of dictionaries containing information about each frame

    Parameter root: root path of KITTI dataset
    Parameter mode: training, testing, or validation mode
    Return: a dictionary containing information about the entire dataset
    """
    dataset_info = {}
    frame_ids = [line.strip() for line in open(os.path.join(root, mode + '.txt'), 'r')]
    for id in tqdm(frame_ids):
        frame_info = {}
        image_path = os.path.join(root, mode, 'images', id + '.png')
        velodyne_path = os.path.join(root, mode, 'velodyne', id + '.bin')
        calib_path = os.path.join(root, mode, 'calib', id + '.txt')
        labels_path = os.path.join(root, mode, 'labels', id + '.txt')

        frame_info['velodyne_path'] = velodyne_path
        frame_info['calib'] = process_calib(calib_path)
        
        image = cv2.imread(image_path)
        image_shape = image.shape[:2]
        frame_info['image'] = {
            'image_shape': image_shape,
            'image_path': image_path,
        }

        reduced_lidar_pc = process_lidar(velodyne_path, frame_info['calib']['R0_rect'], frame_info['calib']['Tr_velo_to_cam'], frame_info['calib']['P2'], image_shape)
        velodyne_reduced_path = os.path.join(root, mode, 'velodyne_reduced', id + '.bin')
        with open(velodyne_reduced_path, 'w') as f:
            reduced_lidar_pc.tofile(f)

        labels = process_labels(labels_path)
        frame_info['labels'] = labels
        dataset_info[int(id)] = frame_info
    with open(os.path.join(root, f'dataset_{mode}_info.pkl'), 'wb') as f:
        pickle.dump(dataset_info, f)
    return dataset_info

if __name__ == "__main__":
    root = '/Volumes/G-DRIVE mobile/kitti'
    mode = 'training'
    generate_dataset_info(root, mode)