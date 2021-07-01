import sys
sys.path.append('../src')
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_anchors, decode_boxes

def kl_divergence_loss(logits, target):
    T = 0.01
    alpha = 0.6
    thresh = 100
    criterion = nn.MSELoss()
    # c : preprocess for distillation
    log2div = logits[1].clamp(-thresh, thresh).sigmoid().squeeze(dim=-1)
    tar2div = target[1].clamp(-thresh, thresh).sigmoid().squeeze(dim=-1).detach()
    closs = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((log2div / T), dim = 1), F.softmax((tar2div / T), dim = 1))*(alpha * T * T) + F.binary_cross_entropy(log2div, tar2div) * (1-alpha)
    
    # r
    anchor = load_anchors("src/anchors.npy")
    rlogits = decode_boxes(logits[0], anchor)
    rtarget = decode_boxes(target[0], anchor)
    rloss = criterion(rlogits, rtarget) 
     
    return closs + rloss


def alternative_kl_loss(logits, target):
    T = 0.01
    alpha = 0.6
    thresh = 100
    criterion = nn.L1Loss()
    # c : preprocess for distillation
    log2div = logits[1].clamp(-thresh, thresh).sigmoid().squeeze(dim=-1)
    tar2div = target[1].clamp(-thresh, thresh).sigmoid().squeeze(dim=-1)
    closs = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((log2div / T), dim = 1), F.softmax((tar2div / T), dim = 1))*(alpha * T * T) + criterion(log2div, tar2div) * (1-alpha)
    
    # r
    anchor = load_anchors("src/anchors.npy")
    rlogits = decode_boxes(logits[0], anchor)
    rtarget = decode_boxes(target[0], anchor)
    rloss = criterion(rlogits, rtarget)
     
    return closs + rloss
