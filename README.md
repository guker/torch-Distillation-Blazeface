# torch-BlazefaceDistillation


# Abstract
This is Distillation for Converting performance of [MobileNet backborn BlazeFace(google)](https://google.github.io/mediapipe/solutions/face_detection.html) to ResNet backborn BlazeFace(original).

<img src="https://user-images.githubusercontent.com/48679574/122851087-5bbb3180-d349-11eb-8cda-82ff78a8efb4.png" width="700px">


Use KL divergence loss and softmax with temperature.
```
def kl_divergence_loss(logits, target):
    T = 0.01
    alpha = 0.6
    criterion = nn.SmoothL1Loss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((logits[0] / T), dim = 1), F.softmax((target[0] / T), dim = 1))*(alpha * T * T) + criterion(logits[0], target[0]) * (1-alpha)

    return kl_loss 
```



# Distillation perfomance (Resnet backborn BlazeFace)

<b>1.MobileNet-backborn.(left)   Resnet-backborn(right)</b>





<b>2.MobileNet-backborn.(left)   Resnet-backborn(right)</b>


# training log 


