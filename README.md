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
    criterion = nn.SmoothL1Loss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((logits[0] / T), dim = 1), F.softmax((target[0] / T), dim = 1))*(alpha * T * T) + criterion(logits[0], target[0]) * (1-alpha)

    return kl_loss 
```



## Distillation perfomance (Resnet backborn BlazeFace)

<b>1.MobileNet-backborn.(left)   Resnet-backborn(right)</b>


<img src="https://user-images.githubusercontent.com/48679574/123735333-fb446b00-d8d9-11eb-8e35-d693082ada66.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/123735337-fd0e2e80-d8d9-11eb-8135-44d646f9e5f9.png" width="400px">






<b>2.MobileNet-backborn.(left)   Resnet-backborn(right)</b>

<img src="https://user-images.githubusercontent.com/48679574/123735525-55ddc700-d8da-11eb-86c9-68f31b4b7d30.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/123735522-55453080-d8da-11eb-833d-e11411159e53.png" width="400px">


## training log 

<b>loss curve</b>

<img src="https://user-images.githubusercontent.com/48679574/124040876-ba1a9b00-da40-11eb-99f7-6097a6b2e1f1.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/124040868-b6871400-da40-11eb-8453-fccfbbb90cb4.png" width="400px">


<b>2 output mae accuracy</b>

<img src="https://user-images.githubusercontent.com/48679574/124040879-bb4bc800-da40-11eb-98f8-666461d94d5f.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/124040881-bbe45e80-da40-11eb-9beb-6de8ee746e29.png" width="400px">



# References
・[MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch)

・[tf-blazeface](https://github.com/FurkanOM/tf-blazeface)
