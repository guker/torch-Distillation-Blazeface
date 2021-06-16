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

ANCHOR_PATH = "src/anchors.npy"    
    
def load_blazeface_net(device, teacher=False):
    front_net = BlazeFace().to(device)
    if teacher:
        front_net.load_weights("src/blazeface.pth")
    # Optionally change the thresholds:
    front_net.min_score_thresh = 0.75
    front_net.min_suppression_threshold = 0.3
    return front_net


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
        return img / 127.5 - 1.0

    def __len__(self):
        return len(self.images)
        
    def _preprocess(self, x):
        return x / 127.5 - 1.0

def get_dataset(config):
    """ Dataset And Augmentation
    """
    train_transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
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
    
    return train_dst, val_dst
   
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
    temperature = 10
    lambda_factor = 1
    rkl_loss = F.kl_div(F.log_softmax((logits[0] / temperature), dim = 0), F.softmax((target[0] / temperature), dim = 0),
                                reduction="batchmean")

    ckl_loss = F.kl_div(F.log_softmax((logits[1] / temperature), dim = 0), F.softmax((target[1] / temperature), dim = 0),
                                reduction="batchmean").mean()
    kl_loss = rkl_loss + ckl_loss
    return kl_loss/2

def train(config, device, teacher_net, student_net, num_workers, epochs=10):
    # config
    batch_size = config.batch_size
    test_batch_size = config.val_batch_size
    lr = config.lr
    eta_min = config.eta_min
    t_max = config.t_max
    val_interval = config.val_interval
    temperature = config.temperature
    lambda_factor = config.lambda_factor

    print('loading dataloader....')
    train_dst, val_dst = get_dataset(config)
    train_loader = call_data_loader(train_dst, bs=config.batch_size, num_worker=num_workers)
    val_loader = call_data_loader(val_dst, bs=config.val_batch_size, num_worker=num_workers)
    
    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}')
    
    print('load mdoel && set parameter')
    npy_anchors = load_anchors(device, path=ANCHOR_PATH)
    front_net = load_blazeface_net(device, teacher=False)
        
    model = student_net
    optimizer, scheduler = create_optimizer(model, config)
    global_step = 0
    c = 1
    mb = master_bar(range(epochs))
    for epoch in mb:
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for i, images in enumerate(progress_bar(train_loader)):
            global_step += 1
            x_batch = images.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            lesson_output = teacher_net(x_batch)
            logits = model(x_batch)
            #lessen_detections = mean(front_net._tensors_to_detections(lesson_output[0], lesson_output[1], npy_anchors))
            #logits_detections = mean(front_net._tensors_to_detections(logits[0], logits[1], npy_anchors))
                      
            loss = kl_divergence_loss(logits, lesson_output) 
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)
            if i %20==0:
                print('avg_loss', avg_loss) 
            
            c += 1
            if c %200==0:
                avg_val_loss = 0
                model.eval()
                for idx, val_batch in enumerate(val_loader):
                    val_batch = val_batch.to(device, dtype=torch.float32)
                    ## validation kl_divergence
                    val_lesson = teacher_net(val_batch)
                    val_logits = model(x_batch)
                    #val_lessen_detections = front_net._tensors_to_detections(val_lesson[0], val_lesson[1], npy_anchors)
                #val_logits_detections = front_net._tensors_to_detections(val_logits[0], val_logits[1], npy_anchors)
                    val_loss = kl_divergence_loss(val_logits, val_lesson)
                    avg_val_loss += val_loss.item() / len(val_loader)
                    if idx==100:
                        break
                print('val_loss', avg_val_loss)
                writer.add_scalar('train/avg_Loss', avg_loss, global_step)  
                writer.add_scalar('valid/avg_loss', avg_val_loss, global_step)
                print('{0}/{1}'.format(epoch, epochs), 'avg_loss', avg_loss)
                torch.save(model.state_dict(), 'checkpoints/distillation_ep{0}_totalep{1}.pth'.format(epoch, epochs))
                print('succeess to checkpoints/distillation_ep{0}_totalep{1}.pth'.format(epoch, epochs))
        scheduler.step()
    writer.close()

if __name__=='__main__':
    cfg = Cfg
    os.makedirs(cfg.checkpoints, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    teacher_net = load_blazeface_net(device, teacher=True)
    student_net = load_blazeface_net(device, teacher=False)
    front_net = load_blazeface_net(device, teacher=False)
    train(config=cfg,
          device=device,
          teacher_net = teacher_net,
          student_net = student_net,
          num_workers=0,
          epochs=cfg.TRAIN_EPOCHS)


