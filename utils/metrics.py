import torch

def dice_coeff(pred, target, smooth=1e-6):
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(pred, target, smooth=1e-6):
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def recall_score(pred, target, smooth=1e-6):
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)

def accuracy_score(pred, target):
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    correct = (pred == target).sum()
    return correct.float() / target.numel()
