import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.MobileNetBlazeface import BlazeFace

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
    front_net = BlazeFace().to(device)
    if teacher:
        front_net.load_state_dict(torch.load(weight))
    # Optionally change the thresholds:
    front_net.min_score_thresh = 0.75
    front_net.min_suppression_threshold = 0.3
    return front_net

def _preprocess(x):
    """Converts the image pixels to the range [-1, 1]."""
    return x.float() / 127.5 - 1.0


def load_anchors(device, path="src/anchors.npy"):
    num_anchors = 896
    anchors = torch.tensor(np.load(path), dtype=torch.float32, device=device)
    assert(anchors.ndimension() == 2)
    assert(anchors.shape[0] == num_anchors)
    assert(anchors.shape[1] == 4)
    return anchors

def torch2numpy(x):
    return x.to('cpu').detach().numpy()

def load_images(filenames):
    xfront = np.zeros((len(filenames), 128, 128, 3), dtype=np.uint8)
    for i, filename in enumerate(filenames):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        xfront[i] = cv2.resize(img, (128, 128))
    return xfront


def post_process(front_net, out, anchors_):
    detections = front_net._tensors_to_detections(out[0], out[1], anchors_)
    front_detections = []
    for i in range(len(detections)):
        faces = front_net._weighted_non_max_suppression(detections[i])
        faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17))
        front_detections.append(faces)
    for idx, d in enumerate(front_detections):
          plot_detections(xfront[idx], torch2numpy(front_detections[idx]))


if __name__=='__main__':
    weight = str(sys.argv[1])
    #weight = "src/blazeface.pth"
    filenames = [ "face1.png", "face2.png", "face3.png" ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_net = load_blazeface_net(device, weight, teacher=True)
    student_net = load_blazeface_net(device, teacher=False)
    front_net = load_blazeface_net(device, teacher=False)

    anchors_ = load_anchors(device, path="src/anchors.npy")
    xfront = load_images(filenames)
    x = torch.from_numpy(xfront).permute((0, 3, 1, 2))
    x = x.to(device)
    x = _preprocess(x)
    out = teacher_net(x)
    print(out[0].shape, out[1].shape)
    post_process(front_net, out, anchors_)

