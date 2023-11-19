---
title:  "Pretrained Model"
metadate: "hide"
date : 2023-11-16 18:00:00 +0900
categories: [ ML/DL ]
image: "/assets/images/pretrained-model.png" 
---  

## Pretrained Model 사전학습 모델 사용법

PyTorch 사전 학습 모델 - 이미지 분류 (https://pytorch.org/vision/0.8/models.html)

- **[AlexNet](https://arxiv.org/abs/1404.5997)**
- **[VGG](https://arxiv.org/abs/1409.1556)**
- **[ResNet](https://arxiv.org/abs/1512.03385)**
- **[SqueezeNet](https://arxiv.org/abs/1602.07360)**
- **[DenseNet](https://arxiv.org/abs/1608.06993)**
- **[Inception](https://arxiv.org/abs/1512.00567) v3**
- **[GoogLeNet](https://arxiv.org/abs/1409.4842)**
- **[ShuffleNet](https://arxiv.org/abs/1807.11164) v2**
- **[MobileNet](https://arxiv.org/abs/1801.04381) v2**
- **[ResNeXt](https://arxiv.org/abs/1611.05431)**
- **[Wide ResNet](https://pytorch.org/vision/0.8/models.html#wide-resnet)**
- **[MNASNet](https://arxiv.org/abs/1807.11626)**

```python
from torchvision import models
net = models.resnet18(pretrained = True)
```

### Fine Tuning 파인 튜닝

Fine Tuning이란 사전 학습 모델의 파라미터를 초깃값으로 사용하지만, 모든 레이어 함수를 재학습시킨다.

```python
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# 난수 고정
torch_seed()

# 최종 노드의 출력을 2로 변경
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))
```

### Transfer Learning 전이 학습

Transfer Learning 전이학습이란 사전 학습 모델의 파라미터 중에서 input layer에 가까운 layer는 그대로 두고, output layer를 원하는 데이터셋에 맞게 교체한다. input layer에 가까운 layer의 파라미터는 재학습 시키지 않고, output layer의 파라미터만 학습시킨다.

```python
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# 모든 파라미터의 경사 계산을 OFF로 설정
for param in net.parameters():
    param.requires_grad = False

# 난수 고정
torch_seed()

# 최종 노드의 출력을 2로 변경
# 이 노드에 대해서만 경사 계산을 수행하게 됨
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수로 교차 엔트로피 사용
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
# 파라미터 수정 대상을 최종 노드로 제한
optimizer = optim.SGD(net.classifier[6].parameters(),lr=lr,momentum=0.9)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))
```

> 해당 포스팅은 '차근차근 실습하며 배우는 파이토치 딥러닝 프로그래밍'의 Ch11을 참고하여 작성되었습니다.