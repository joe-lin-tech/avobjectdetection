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
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
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
            cls_pred, reg_pred, dir_pred, anchor_match = point_pillars(
                data['batched_pointclouds'], data['batched_gt_boxes'], data['batched_gt_labels'])
            # calculate loss
            batched_cls = anchor_match['batched_cls'].reshape(-1)
            batched_reg = anchor_match['batched_reg'].reshape(-1, 7)
            batched_dir = anchor_match['batched_dir'].reshape(-1)
            batched_ambiguity = anchor_match['batched_ambiguity'].reshape(-1)

            pos_flag = (batched_cls >= 0) & (batched_cls < 3)
            reg_pred = reg_pred[pos_flag]
            batched_reg = batched_reg[pos_flag]
            dir_pred = dir_pred[pos_flag]
            batched_dir = batched_dir[pos_flag]
            # set unobserved classes to num_classes
            batched_cls[batched_cls < 0] = 3
            batched_cls = batched_cls[batched_ambiguity == 0]
            cls_pred = cls_pred[batched_ambiguity == 0]

            loss_values = loss(cls_pred, reg_pred, dir_pred,
                               batched_cls, batched_reg, batched_dir)
            # backward pass
            total_loss = loss_values['total_loss']
            total_loss.backward()
            # update weights
            optim.step()
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Iteration {i + 1}/{num_iters}, Loss: {loss_values["total_loss"]:.4f}, Classification Loss: {loss_values["cls_loss"]:.4f}, Regression Loss: {loss_values["reg_loss"]:.4f}, Direction Loss: {loss_values["dir_loss"]:.4f}')
            if i == num_iters:
                break
        # scheduler.step()
    return


if __name__ == "__main__":
    root = '/Volumes/G-DRIVE mobile/kitti'
    mode = 'training'
    dataset = train_model(root, 1, 10, 2)
