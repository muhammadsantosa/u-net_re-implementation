import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, omega1=1.0, omega2=1.0, epsilon=1e-6):

        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.omega1 = omega1
        self.omega2 = omega2
        self.epsilon = epsilon

    def forward(self, preds, targets):
        preds = preds.contiguous().view(preds.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        preds_sigmoid = torch.sigmoid(preds)
        pt = preds_sigmoid * targets + (1 - preds_sigmoid) * (1 - targets) 
        focal_loss = -self.alpha * (1 - pt).pow(self.gamma) * torch.log(pt + self.epsilon)
        focal_loss = focal_loss.mean()

        intersection = (preds_sigmoid * targets).sum(dim=1)
        union = preds_sigmoid.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1.0 - dice.mean()

        log_cosh_dice_loss = torch.log(torch.cosh(dice_loss))

        combined_loss = self.omega1 * focal_loss + self.omega2 * (dice_loss**self.gamma) * log_cosh_dice_loss
        return combined_loss