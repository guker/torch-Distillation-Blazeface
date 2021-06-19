import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.MobileNetBlazeface import BlazeFace

def load_blazeface_net(device):
    front_net = BlazeFace().to(device)
    front_net.load_weights("src/blazeface.pth")
    front_net.load_anchors("src/anchors.npy")

    # Optionally change the thresholds:
    front_net.min_score_thresh = 0.75
    front_net.min_suppression_threshold = 0.3
    return front_net



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


def single_detect(path="face1.png"):
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (128, 128))
  front_detections = front_net.predict_on_image(img)
  plot_detections(img, front_detections)


def multi_detect(filenames):
  xfront = np.zeros((len(filenames), 128, 128, 3), dtype=np.uint8)
  xback = np.zeros((len(filenames), 256, 256, 3), dtype=np.uint8)

  for i, filename in enumerate(filenames):
      img = cv2.imread(filename)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      xfront[i] = cv2.resize(img, (128, 128))
      xback[i] = cv2.resize(img, (256, 256))
  front_detections = front_net.predict_on_batch(xfront)
  for idx, d in enumerate(front_detections):
      plot_detections(xfront[idx], front_detections[idx])


def main(filenames):
  print("PyTorch version:", torch.__version__)
  print("CUDA version:", torch.version.cuda)
  print("cuDNN version:", torch.backends.cudnn.version())
  #single_detect(path=filenames[0])
  multi_detect(filenames)

if __name__=='__main__':
  filenames = [ "face1.png", "face2.png", "face3.png" ]
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  front_net = load_blazeface_net(device)
  print(front_net)
  main(filenames)
