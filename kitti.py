import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle
from utils import camera_to_lidar

class KITTIDataset(Dataset):
    """
    A class representing the KITTI dataset for 3D object detection
    
    Attribute root: root path of KITTI dataset
    Attribute mode: training, testing, or validation mode
    Atrribute frames_info: a list of dictionaries containing information about each frame
    """
    def __init__(self, root, mode):
        """
        Initializing KITTIDataset object

        Parameter root: root path of KITTI dataset
        Parameter mode: training, testing, or validation mode
        """
        self.root = root
        self.mode = mode
        with open(os.path.join(root, f'dataset_{mode}_info.pkl'), 'rb') as f:
            self.dataset_info = pickle.load(f)
        self.classes = {
            'Pedestrian': 0,
            'Cyclist': 1,
            'Car': 2,
        }

    def __getitem__(self, index):
        """
        Get a frame from the dataset

        Parameter index: index of the frame
        Return: a dictionary containing the frame data
        """
        frame_info = self.dataset_info[index]
        image_path = frame_info['image_path']
        
        # load pointcloud data
        velodyne_path = frame_info['velodyne_path']
        lidar_pc = np.reshape(np.fromfile(velodyne_path, dtype=np.float32), (-1, 4))

        # load calibration data
        calib = frame_info['calib']

        # load label data
        labels = frame_info['labels']
        gt_boxes = np.hstack((labels['location'], labels['dimensions'], np.reshape(labels['rotation_y'], (-1, 1))))
        gt_boxes = camera_to_lidar(gt_boxes, calib['Tr_velo_to_cam'], calib['R0_rect'])
        gt_labels = np.array([self.classes.get(label, -1) for label in labels['name']])

        frame_dict = {
            'pointcloud': lidar_pc,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'calib': calib,
        }
        return frame_dict

    def __len__(self):
        """
        Get the length of the dataset
            
        Return: the length of the dataset
        """
        return len(self.dataset_info)
