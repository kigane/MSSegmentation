import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight

    def forward(self, inputs, targets, smooth=1e-5):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + self.dice_weight * dice_loss

        return Dice_BCE


class FocalDiceBCELoss(nn.Module):
    def __init__(self, dice_weight, alpha=0.25, gamma=2):
        super(FocalDiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1e-5):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(
            inputs, targets.float(), reduction='none')  # -logp
        p = torch.exp(-BCE)
        alpha_lst = torch.tensor(
            [1-self.alpha, self.alpha], device=inputs.device)
        alpha = alpha_lst[targets.data.view(-1).long()]
        loss = alpha * (1-p)**self.gamma * BCE
        Dice_BCE = loss.mean() + self.dice_weight * dice_loss

        return Dice_BCE


if __name__ == '__main__':
    x = torch.randint(2, (3, 6)).float()
    y = torch.randint(2, (3, 6))
    l = FocalDiceBCELoss(0)
    print(l(x, y))
