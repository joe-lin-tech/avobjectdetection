import os
from tqdm import tqdm
import numpy as np
import pickle

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

        frame_info['image_path'] = image_path
        frame_info['velodyne_path'] = velodyne_path
        frame_info['calib'] = process_calib(calib_path)
        
        lidar_pc = np.reshape(np.fromfile(velodyne_path, dtype=np.float32), (-1, 4))
        # TODO remove lidar points that are outside of camera view

        labels = process_labels(labels_path)
        frame_info['labels'] = labels
        dataset_info[int(id)] = frame_info
    with open(os.path.join(root, f'dataset_{mode}_info.pkl'), 'wb') as f:
        pickle.dump(dataset_info, f)
    return dataset_info

if __name__ == "__main__":
    root = '/Volumes/G-DRIVE mobile/kitti'
    mode = 'training'
    dataset = generate_dataset_info(root, mode)