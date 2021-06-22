import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_loss(logits, target):
    T = 0.01
    alpha = 0.6
    K = 1
    criterion = nn.SmoothL1Loss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((logits[0] / T), dim = 1), F.softmax((target[0] / T), dim = 1))*(alpha * T * T) + criterion(logits[0], target[0]) * (1-alpha)

    return kl_loss * K
