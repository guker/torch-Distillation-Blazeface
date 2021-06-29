# torch-BlazefaceDistillation


# Abstract
This is Distillation for Converting performance of [MobileNet backborn BlazeFace(google)](https://google.github.io/mediapipe/solutions/face_detection.html) to ResNet backborn BlazeFace(original).

<b>Distillation sturacture</b>

<img src="https://user-images.githubusercontent.com/48679574/123763800-1ecddc80-d8ff-11eb-93ed-d1cb89d20295.png" width="700px">


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

<img src="https://user-images.githubusercontent.com/48679574/123735623-84f43880-d8da-11eb-8fba-980765647bbf.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/123735625-86256580-d8da-11eb-92ea-caca8d2f7594.png" width="400px">


<b>mae accuracy</b>

<img src="https://user-images.githubusercontent.com/48679574/123735631-87569280-d8da-11eb-809e-a4be5611d57e.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/123735633-8887bf80-d8da-11eb-8ebc-9db46cfd7a09.png" width="400px">



# References
・[MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch)

・[tf-blazeface](https://github.com/FurkanOM/tf-blazeface)
