# torch-BlazefaceDistillation


# Abstract
This is Distillation for Converting performance of [MobileNet backborn BlazeFace(google)](https://google.github.io/mediapipe/solutions/face_detection.html) to ResNet backborn BlazeFace(original).

<b>Distillation sturacture</b>

<img src="https://user-images.githubusercontent.com/48679574/124056945-9ae03580-da61-11eb-850c-3bee99724ef2.png" width="700px">


Use KL divergence loss and softmax with temperature.
```python
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
```



## Distillation perfomance (Resnet backborn BlazeFace)

<b>1.MobileNet-backborn (google pretrained tflite model)</b>

<img src="https://user-images.githubusercontent.com/48679574/124057053-bea37b80-da61-11eb-9101-2b5b98081a66.png" width="450px"><img src="https://user-images.githubusercontent.com/48679574/124057045-bd724e80-da61-11eb-80f1-67572ea7ae45.png" width="450px">




<b>2.Resnet-backborn(Distillation customized Blazeface)</b>

<img src="https://user-images.githubusercontent.com/48679574/124057242-0d511580-da62-11eb-8f6b-3f47ed28ef63.png" width="450px"><img src="https://user-images.githubusercontent.com/48679574/124057239-0c1fe880-da62-11eb-99d7-c4a77e8b783c.png" width="450px">


## training log 

<b>loss curve</b>

<img src="https://user-images.githubusercontent.com/48679574/124040876-ba1a9b00-da40-11eb-99f7-6097a6b2e1f1.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/124040868-b6871400-da40-11eb-8453-fccfbbb90cb4.png" width="400px">


<b>2 output mae accuracy</b>

<img src="https://user-images.githubusercontent.com/48679574/124040879-bb4bc800-da40-11eb-98f8-666461d94d5f.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/124040881-bbe45e80-da40-11eb-9beb-6de8ee746e29.png" width="400px">



# References
・[MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch)

・[tf-blazeface](https://github.com/FurkanOM/tf-blazeface)
