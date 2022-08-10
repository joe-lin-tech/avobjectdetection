from kitti import KITTIDataset
from torch.utils.data import DataLoader
from utils import collate_fn
from model import PointPillars
from loss import PointPillarsLoss
import torch
from tqdm import tqdm

def train_model(root, num_epochs, num_iters, batch_size):
    """
    Train PointPillars model on the KITTI dataset

    Parameter root: root path of KITTI dataset=
    Parameter num_epochs: number of epochs to train
    Parameter num_iters: number of iterations per epoch
    Parameter batch_size: batch size
    """
    # initialize dataset and dataloaders
    train_dataset = KITTIDataset(root, 'training')
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    # validation_dataset = KITTIDataset(root, 'validation')
    # test_dataset = KITTIDataset(root, 'testing')
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # check if M1 GPU is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    # initialize model, optimizer, scheduler, and loss function
    point_pillars = PointPillars(
        voxel_size=[0.16, 0.16, 4],
        pointcloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        max_num_points=32,
        max_voxels=16000)
    # TODO reactivate mps when operations are compatible
    # point_pillars.to("mps")
    optim = torch.optim.Adam(point_pillars.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    loss = PointPillarsLoss(cls_beta=1.0, reg_beta=2.0, dir_beta=0.2)

    # training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(train_dataloader, desc='Epoch ' + str(epoch))):
            # reset gradients
            optim.zero_grad()
            # forward pass
            output = point_pillars(data['batched_pointclouds'])
            # calculate loss
            loss_value = loss(output, data['labels'])
            # backward pass
            loss_value.backward()
            # update weights
            optim.step()
            print(f'Epoch {epoch + 1}/{num_epochs}, Iteration {i + 1}/{num_iters}, Loss: {loss_value.item():.4f}')
            if i == num_iters:
                break
        # scheduler.step()
    return


if __name__ == "__main__":
    root = '/Volumes/G-DRIVE mobile/kitti'
    mode = 'training'
    dataset = train_model(root, 1, 10, 2)
