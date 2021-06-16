




def train_step(x, t):
    model.train()
    preds = model(x)
    loss = compute_loss(t, preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, preds

def test_step(x, t):
    model.eval()
    preds = model(x)
    loss = compute_loss(t, preds)
    return loss, preds

np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataloader, test_dataloader = call_data_loader()
model = ResNet50(10).to(device)
print(model)

criterion = nn.NLLLoss()
optimizer = optimizers.Adam(model.parameters(), weight_decay=0.01)
epochs = 5
for epoch in range(epochs):
    train_loss = 0.
    test_loss = 0.
    test_acc = 0.
    for (x, t) in train_dataloader:
        x, t = x.to(device), t.to(device)
        loss, _ = train_step(x, t)
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    for (x, t) in test_dataloader:
        x, t = x.to(device), t.to(device)
        loss, preds = test_step(x, t)
        test_loss += loss.item()
        test_acc += \
            accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
        epoch+1,
        test_loss,
        test_acc
    ))




#######

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.utils.data import Dataset, DataLoader
import torchvision
from cfg import Cfg
from blazeface import BlazeFace

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    args = vars(parser.parse_args())
    cfg.update(args)
    return edict(cfg)

def load_teacher_model(device):
    front_net = BlazeFace().to(device)
    front_net.load_weights("blazeface.pth")
    front_net.load_anchors("anchors.npy")
    back_net = BlazeFace(back_model=True).to(device)
    back_net.load_weights("blazefaceback.pth")
    back_net.load_anchors("anchorsback.npy")

    # Optionally change the thresholds:
    front_net.min_score_thresh = 0.75
    front_net.min_suppression_threshold = 0.3
    return front_net
    
    
def load_student_model(device):
    front_net = BlazeFace().to(device)
    #front_net.load_weights("blazeface.pth")
    front_net.load_anchors("anchors.npy")
    back_net = BlazeFace(back_model=True).to(device)
    #back_net.load_weights("blazefaceback.pth")
    back_net.load_anchors("anchorsback.npy")

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
        
        self.transform = transform
        self.width = width
        self.height = height
        train_img_dir = os.listdir(image_dir)
        train_img_dir.sort()
        # jpg image
        self.images = [os.path.join(image_dir, path) for path in train_img_dir]

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.height, self.width), cv2.INTER_NEAREST)
        img = img.astype(np.float32)/255
        
        if self.transform is not None:
            augment = self.transform(image=img)
            img = augment['image']
        #print(img.numpy().shape, target.numpy().shape)
        return img


def get_dataset(config, width=int(1216/4), height=int(1936/4)):
    """ Dataset And Augmentation
    """
    train_transform = albu.Compose([
            albu.HorizontalFlip(p=1),
            ToTensorV2(),
            ])

    val_transform = albu.Compose([
        ToTensorV2(),
        ])

    train_dst = DataLoader(image_dir=, width=width, height=height, transform=train_transform)
    val_dst = DataLoader(image_dir=, width=width, height=height, transform=val_transform)
    
    return train_dst, val_dst
   
def call_data_loader(dst, bs=4, shuffle=True, num_worker=0):
    loader = data.DataLoader(
        dst, batch_size=bs, shuffle=True, num_workers=num_worker)
    return loader
   
   
   
def kl_divergence_loss(criterion, preds, lesson):
    temperature = 10
    lambda_factor=1
    kl_loss = F.kl_div(F.log_softmax((preds / temperature), dim = 1), F.softmax((lesson / temperature), dim = 1),
                                reduction="batchmean")
    y_batch = lesson
    loss = criterion(preds, y_batch.cuda()) + lambda_factor * (temperature ** 2) * kl_loss
    return loss
    
def train(config, device, teacher_model, student_model, num_workers):
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    test_batch_size = config
    
    
    print('loading data....')
    train_dst, val_dst = get_dataset(config)
    train_loader = call_data_loader(train_dst, bs=config.batch_size, num_worker=num_workers)
    val_loader = call_data_loader(val_dst, bs=config.val_batch_size, num_worker=num_workers)
    
    lr = config.lr
    eta_min = config.eta_min
    t_max = config.t_max
    
    temperature = config.temperature
    lambda_factor = config.lambda_factor
    criterion = nn.BCEWithLogitsLoss().cuda()
    student_model = student_model.to(device)
    optimizer, scheduler = create_optimizer(student_model, config)
    best_epoch = -1
    best_lwlrap = 0.
    mb = master_bar(range(num_epochs))

    for epoch in mb:
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for images in progress_bar(train_loader):
            images = images.to(device, dtype=torch.float32)
            
            ## pred
            lesson = teacher_model.predict_on_batch(images)
            preds = student_model.predict_on_batch(images)
            loss = kl_divergence_loss(criterion, preds, lesson)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)
    
def train(config, device, teacher_model, student_model):
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    test_batch_size = config
    
    
    print('loading data....')
    train_dst, val_dst = get_dataset(config)
    train_loader = call_data_loader(train_dst, bs=config.train_batch, num_worker=num_workers)
    val_loader = call_data_loader(val_dst, bs=config.val_batch_size, num_worker=num_workers)
    
    lr = config.lr
    eta_min = config.eta_min
    t_max = config.t_max
    
    temperature = config.temperature
    lambda_factor = config.lambda_factor
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    
    best_epoch = -1
    best_lwlrap = 0.
    mb = master_bar(range(num_epochs))

    for epoch in mb:
        start_time = time.time()
        model.train()
        avg_loss = 0.
        
        for x_batch in progress_bar(train_loader):
            teacher1 = model1(x_batch[:, 0, :, :].view(-1, 1, 299, 299).cuda())
            teacher3 = model3(x_batch.cuda())
            lesson = ((teacher1 + teacher3) / 2)
            
            preds = model(x_batch.cuda())
            
            kl_loss = F.kl_div(F.log_softmax((preds / temperature)), F.softmax((lesson / temperature)),
                               reduction="batchmean")

            # loss = criterion(preds, y_batch.cuda())
            # loss = F.cross_entropy(preds, y_batch.cuda()) + lambda_factor * (temperature ** 2) * kl_loss
            loss = criterion(preds, y_batch.cuda()) + lambda_factor * (temperature ** 2) * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)
        
        model.eval()
        valid_preds = np.zeros((len(x_val), num_classes))
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            preds = model(x_batch.cuda()).detach()
            loss = criterion(preds, y_batch.cuda())

            preds = torch.sigmoid(preds)
            valid_preds[i * test_batch_size: (i+1) * test_batch_size] = preds.cpu().numpy()

            avg_val_loss += loss.item() / len(valid_loader)
            
        score, weight = calculate_per_class_lwlrap(y_val, valid_preds)
        lwlrap = (score * weight).sum()
        
        scheduler.step()

        if (epoch + 1) % 2 == 0:
            elapsed = time.time() - start_time
            mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  val_lwlrap: {lwlrap:.6f}  time: {elapsed:.0f}s')
    
        if lwlrap > best_lwlrap:
            best_epoch = epoch + 1
            best_lwlrap = lwlrap
            torch.save(model.state_dict(), 'weight_best.pt')
            
    return {
        'best_epoch': best_epoch,
        'best_lwlrap': best_lwlrap,
    }
    
    
if __name__=='__main__':
    cfg = get_args(**Cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_model = load_teacher_model(device)
    student_model = load_student_model(device)
    train(config=cfg,
          device=device,
          teacher_model = teacher_model,
          student_model = student_model)
    
