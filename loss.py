import torch
import torch.nn as nn
import torch.nn.functional as F

class PointPillarsLoss(nn.Module):
    """
    A class representing the loss function of PointPillars

    Attribute cls_beta: beta coefficient of classification loss
    Parameter reg_beta: beta coefficient of regression loss
    Parameter dir_beta: beta coefficient of direction loss
    """
    def __init__(self, cls_beta, reg_beta, dir_beta):
        super().__init__()
        self.alpha = 0.25
        self.beta = 1/9
        self.gamma = 2.0
        self.cls_beta = cls_beta
        self.reg_beta = reg_beta
        self.dir_beta = dir_beta


    def forward(self, cls_pred, reg_pred, dir_pred, batched_cls, batched_reg, batched_dir):
        """
        Compute the loss
        
        Parameter cls_pred: predicted classification scores
        Parameter reg_pred: predicted regression values
        Parameter dir_pred: predicted direction values
        Parameter batched_cls: ground truth classification labels
        Parameter batched_reg: ground truth regression values
        Parameter batched_dir: ground truth direction values
        Return: loss value
        """
        # classification loss
        # create one hot encoding of ground truth labels with num_classes + 1
        num_cls_pos = (batched_cls < 3).sum()
        print(num_cls_pos)
        batched_cls = F.one_hot(batched_cls, num_classes=4)[:, :3].float()
        cls_sigmoid = torch.sigmoid(cls_pred)
        cls_weights = self.alpha * (1 - cls_sigmoid).pow(self.gamma) * batched_cls + \
            (1 - self.alpha) * cls_sigmoid.pow(self.gamma) * (1 - batched_cls)
        cls_loss = F.binary_cross_entropy(cls_sigmoid, batched_cls, reduction='none')
        cls_loss = cls_loss * cls_weights
        cls_loss = cls_loss.sum() / num_cls_pos

        # regression loss
        reg_loss = F.smooth_l1_loss(reg_pred, batched_reg, reduction='mean', beta=self.beta)

        # direction loss
        dir_loss = F.cross_entropy(dir_pred, batched_dir, reduction='mean')
        
        # total loss
        total_loss = self.cls_beta * cls_loss + self.reg_beta * reg_loss + self.dir_beta * dir_loss

        loss_values = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'dir_loss': dir_loss,
            'total_loss': total_loss,
        }
        return loss_values