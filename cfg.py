import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.batch_size = 4
Cfg.subdivisions=1
Cfg.val_batch_size = 4
Cfg.width = 128
Cfg.height = 128
Cfg.lr = 0.001
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.eta_min = 1e-5
Cfg.t_max = 10
Cfg.temperature = 10
Cfg.lambda_factor=1


Cfg.TRAIN_EPOCHS = 300
Cfg.val_interval = 1
Cfg.gpu_id = '0'
Cfg.train_img_dir = '../UTKDATA/UTKFace'
Cfg.valid_mask_dir = '../UTKDATA/part3'
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')
Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.keep_checkpoint_max = 10
