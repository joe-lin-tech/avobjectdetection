from kitti import KITTIDataset
from torch.utils.data import DataLoader
from utils import collate_fn
from model import PointPillars

def train_model(root):
    """
    Train PointPillars model on the KITTI dataset

    Parameter root: root path of KITTI dataset
    Parameter model: PointPillars model
    Parameter optimizer: optimizer for training
    Parameter scheduler: learning rate scheduler
    Parameter num_epochs: number of epochs to train
    """
    # initialize dataset and dataloaders
    train_dataset = KITTIDataset(root, 'training')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    # validation_dataset = KITTIDataset(root, 'validation')

    point_pillars = PointPillars()

    return

if __name__ == "__main__":
    root = '/Volumes/G-DRIVE mobile/kitti'
    mode = 'training'
    dataset = train_model(root)