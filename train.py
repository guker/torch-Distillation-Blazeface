import os, cv2, sys
import argparse
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optimizers
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from fastprogress import master_bar, progress_bar

from torch.optim.lr_scheduler import CosineAnnealingLR
from cfg import Cfg
from models.MobileNetBlazeface import BlazeFace
from models.ResnetBlazeface import BlazeFace as resnetBlazeFace

ANCHOR_PATH = "src/anchors.npy"    
    
def load_blazeface_net(device, weights=None, teacher=False):
    student_net = resnetBlazeFace().to(device)
    if weights:
        student_net.load_state_dict(torch.load(weights))
    if teacher:
        teacher_net = BlazeFace().to(device)
        teacher_net.load_weights("src/blazeface.pth")
        teacher_net.min_score_thresh = 0.75
        teacher_net.min_suppression_threshold = 0.3
        return teacher_net
    # Optionally change the thresholds:
    student_net.min_score_thresh = 0.75
    student_net.min_suppression_threshold = 0.3
    return student_net


class DataLoader(data.Dataset):
    def __init__(self,
                 image_dir,
                 width,
                 height,
                 transform=None):
        self.width = width
        self.height = height
        train_img_dir = os.listdir(image_dir)
        train_img_dir.sort()
        # jpg image
        self.images = [os.path.join(image_dir, path) for path in train_img_dir]
        self.transforms = transform
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = img.resize((self.height, self.width))
        #img = self._preprocess(img)
        img = self.transforms(img)
        return img / 255

    def __len__(self):
        return len(self.images)
        
    def _preprocess(self, x):
        return x / 255

def get_dataset(config):
    """ Dataset And Augmentation
    """
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    train_dst = DataLoader(image_dir=config.train_img_dir, width=config.width,
                           height=config.height, transform=train_transform)
    val_dst = DataLoader(image_dir=config.valid_mask_dir, width=config.width,
                         height=config.height, transform=val_transform)
    test_dst = DataLoader(image_dir=config.test_dir, width=config.width,
                         height=config.height, transform=val_transform)
    return train_dst, val_dst, test_dst
   
def call_data_loader(dst, bs=4, shuffle=True, num_worker=0):
    loader = data.DataLoader(
        dst, batch_size=bs, shuffle=True, num_workers=num_worker)
    return loader


def create_optimizer(model, config):
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(params=[
        {'params': model.parameters(), 'lr': config.lr},
        ], lr=config.lr, betas=(0.9, 0.999), eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=[
        {'params': model.parameters(), 'lr': config.lr},
        ], lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.t_max, eta_min=config.eta_min)
    return optimizer, scheduler

def load_anchors(device, path=ANCHOR_PATH):
    num_anchors = 896
    anchors = torch.tensor(np.load(path), dtype=torch.float32, device=device)
    assert(anchors.ndimension() == 2)
    assert(anchors.shape[0] == num_anchors)
    assert(anchors.shape[1] == 4)
    return anchors
    
    
def kl_divergence_loss(logits, target):
    T = 0.01
    alpha = 0.6
    K = 1
    criterion = nn.L1Loss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((logits[0] / T), dim = 1), F.softmax((target[0] / T), dim = 1))*(alpha * T * T) + criterion(logits[0], target[0]) * (1-alpha)
    return kl_loss * K

def train(config, device, teacher_net, student_net, num_workers, epochs=10):
    # config
    batch_size = config.batch_size
    test_batch_size = config.val_batch_size
    lr = config.lr
    eta_min = config.eta_min
    t_max = config.t_max

    print('loading dataloader....')
    train_dst, val_dst, test_dst = get_dataset(config)
    train_loader = call_data_loader(train_dst, bs=config.batch_size, num_worker=num_workers)
    val_loader = call_data_loader(val_dst, bs=config.val_batch_size, num_worker=num_workers)
    test_loader = call_data_loader(test_dst, bs=1, num_worker=num_workers)
    
    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}')
    
    print('load mdoel && set parameter')
    npy_anchors = load_anchors(device, path=ANCHOR_PATH)
    front_net = load_blazeface_net(device, teacher=False)
    
    acc_criterion = nn.L1Loss()    
    model = student_net
    optimizer, scheduler = create_optimizer(model, config)
    cur_itrs = 0
    total_itrs = config.total_itrs

    while cur_itrs < total_itrs:
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for images in progress_bar(train_loader):
            cur_itrs += 1
            x_batch = images.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            lesson = teacher_net(x_batch)
            logits = model(x_batch)
            
            loss = kl_divergence_loss(logits, lesson) 
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() 
            if cur_itrs % 100==0:
                print('avg_loss', avg_loss/100)
                writer.add_scalar('train/avg_Loss', avg_loss, cur_itrs) 
                avg_loss = 0.
          
            if cur_itrs %2000==0:
                avg_val_loss = 0
                model.eval()
                for idx, val_batch in tqdm(enumerate(val_loader)):
                    val_batch = val_batch.to(device, dtype=torch.float32)
                    ## validation kl_divergence
                    val_lesson = teacher_net(val_batch)
                    val_logits = model(val_batch)
                    val_loss = kl_divergence_loss(val_logits, val_lesson)
                    avg_val_loss += val_loss.item() / len(val_loader)
                     
                print('val_loss', avg_val_loss)
                writer.add_scalar('valid/avg_loss', avg_val_loss, cur_itrs)
                
                torch.save(model.state_dict(), 'checkpoints/student_iter{}.pth'.format(cur_itrs))
                print('succeess to checkpoints/student_iter{}.pth'.format(cur_itrs))
                model.train()
            
            if cur_itrs %1000==0: 
                model.eval()
                avg_mae_acc = 0
                for test_batch in test_loader:
                    test_batch = test_batch.to(device, dtype=torch.float32)
                    test_lesson = teacher_net(test_batch)
                    test_logits = model(test_batch)
                    #lessen_score = front_net._tensors_to_detections(test_lesson[0], test_lesson[1], npy_anchors, test=True)
                    #logits_score = front_net._tensors_to_detections(test_logits[0], test_logits[1], npy_anchors, test=True)
                    mae_score = acc_criterion(test_logits[0], test_lesson[0])
                    avg_mae_acc += mae_score / len(test_loader)
             
                print('mae accuracy', avg_mae_acc)
                writer.add_scalar('test/avg_mae_acc', avg_mae_acc, cur_itrs)
            
            if cur_itrs > total_itrs:
                break
            model.train()
            scheduler.step()
        writer.close()

if __name__=='__main__':
    cfg = Cfg
    os.makedirs(cfg.checkpoints, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student_weights = 'checkpoints/iter2max600000.pth' 
    teacher_net = load_blazeface_net(device, teacher=True)
    student_net = load_blazeface_net(device, weights=student_weights, teacher=False)
    train(config=cfg,
          device=device,
          teacher_net = teacher_net,
          student_net = student_net,
          num_workers=0)


