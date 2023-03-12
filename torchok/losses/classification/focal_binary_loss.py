import torch
import torch.nn as nn

from torchok.constructor import LOSSES


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
@LOSSES.register_class
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        target = target.float()
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(input, target)
        probas = torch.sigmoid(input)
        loss = target * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - target) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss
