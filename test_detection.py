import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.MobileNetBlazeface import BlazeFace
from models.ResnetBlazeface import BlazeFace as resnetBlazeFace

def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=detections[i, 16])
                ax.add_patch(circle)
    plt.show()


def load_blazeface_net(device, weight=None, teacher=False):
    student_net = resnetBlazeFace().to(device)
    student_net.load_anchors("src/anchors.npy")
    if teacher:
        teacher_net = BlazeFace().to(device)
        teacher_net.load_state_dict(torch.load("src/blazeface.pth"))
        teacher_net.load_anchors("src/anchors.npy")
        teacher_net.min_score_thresh = 0.75
        teacher_net.min_suppression_threshold = 0.3
        return teacher_net
    # Optionally change the thresholds:
    student_net.load_state_dict(torch.load(weight))
    student_net.min_score_thresh = 0.75
    student_net.min_suppression_threshold = 0.3
    return student_net



def load_images(filenames):
    xfront = np.zeros((len(filenames), 128, 128, 3), dtype=np.uint8)
    for i, filename in enumerate(filenames):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        xfront[i] = cv2.resize(img, (128, 128))
    return xfront


def test_detect(front_net, xfront):
  front_detections = front_net.predict_on_batch(xfront, check=True)
  for idx, d in enumerate(front_detections):
      plot_detections(xfront[idx], front_detections[idx])


if __name__=='__main__':
    weight = str(sys.argv[1])
    ver = str(sys.argv[2])
    if ver=='png':
        filenames = [ "face1.png", "face2.png", "face3.png" ]
    else:
        filenames = [ "face1.jpg", "face2.jpg", "face3.jpg" ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_net = load_blazeface_net(device, teacher=True)
    student_net = load_blazeface_net(device, weight=weight, teacher=False)
    
    xfront = load_images(filenames)
    test_detect(teacher_net, xfront)
    test_detect(student_net, xfront)
