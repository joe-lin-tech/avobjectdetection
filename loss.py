import torch
import torch.nn as nn

class PointPillarsLoss(nn.Module):
    """
    A class representing the loss function of PointPillars

    Attribute cls_beta: beta coefficient of classification loss
    Parameter reg_beta: beta coefficient of regression loss
    Parameter dir_beta: beta coefficient of direction loss
    """
    def __init__(self, cls_beta, reg_beta, dir_beta):
        super().__init__()
        self.cls_beta = cls_beta
        self.reg_beta = reg_beta
        self.dir_beta = dir_beta


    def forward(self, cls_pred, reg_pred, dir_pred, batched_labels):
        """
        Compute the loss

        Return: loss value
        """
        return