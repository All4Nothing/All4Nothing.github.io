---
title:  "Prototyping with PyTorch"
metadate: "hide"
date : 2023-12-14 20:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/prototyping-with-pytorch.png" 
---  


## Prototyping with PyTorch

Fast.ai와 PyTorch Lightning은 코드 몇 줄로 빠르게 모델 훈련과 테스트 pipeline을 구축할 수 있는 API를 제공하는 딥러닝 라이브러리이다.

### fast.ai

```python
from fastai.vision.all import *
```

`import *`는 python에서 라이브러리를 import할 때 권장하는 방법은 아니지만, [fast.ai](http://fast.ai)가 사용되게 설계된 REPL(Read-Eval-Print Loop) 환경 때문에 fast.ai 문서(https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/)에서 이 형식을 제안한다.

```python
learn = cnn_learner(dls, arch=resnet18, metrics=accuracy)
```

`fast.ai`의 `cnn_learner` 모듈을 사용해 모델을 인스턴스화 한다. 이때 기본 architecture를 지정해준다. 사용 가능한 기본 architecture의 목록은 https://docs.fast.ai/ 에서 확인할 수 있다.

```python
learn.lr_find()
```

`fast.ai`의 `Learning Rate Finder` 를 사용해 모델 architecture와 데이터셋 조합에 대해 적합한 learning rate를 제안 받을 수 있다. https://docs.fast.ai/callback.schedule.html#LRFinder

```python
learn.fine_tune(epochs=2, base_lr=0.0209, freeze_epochs=1)
```

`Learn.fit`을 사용해 모델을 처음부터 훈련 시킬 수 있지만, `learn.fine_tune`을 통해 사전 훈련된 모델을 미세 조정할 수 있다.

`freeze_epochs`는 초기에 고정 네트워크(마지막 계층을 제외한 모든 계층을 고정)로 모델이 훈련되는 epoch 수를 나타낸다.

`epochs`는 이후 전체 네트워크 계층을 고정하지 않고 모델을 훈련시키는 epoch 수를 나타낸다.

```python
learn.show_results()
```

훈련된 모델을 사용하면 `show_results` 메서드를 사용해 모델 예측의 일부를 볼 수 있다.

```python
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))
```

`fast.ai`의 `Interpretation` 모듈을 사용하면 훈련된 모델이 가장 많이 실패하는 부분을 살펴볼 수 있다. **- Interpretability**

### PyTorch Lightning

https://pytorch-lightning.readthedocs.io/en/stable

PyTorch Lightning은 모델 훈련과 평가에 필요한 boilerplate 코드를 abstract(추상화) 하기 위해 pytorch 위에 구축한 라이브러리이다.

PyTorch Lightning을 사용해 작성된 모델 훈련 코드는 다중 CPU, 다중 GPU, 다중 TPU 처럼 어떤 하드웨어 구성에도 코드를 변경하지 않고 실행할 수 있다.

다양한 모델을 빠르게 실험해보거나 모델 훈련 pipeline에서 scaffolding 코드를 줄일 때 유용하다.

PyTorch Lightning은 **self-contained type(자기완결형)** 모델 시스템의 철학에 따라 작동한다. 즉, 모델 클래스에는 모델 architecture 정의뿐 아니라 optimizer, dataset loader, train, validation, testset 성능 계산 함수 등이 모두 한 곳에 포함되어 있다.

```python
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning as L

class ConvNet(L.LightningModule):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32
        self.fc2 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op

    def training_step(self, batch, batch_num):
        train_x, train_y = batch
        y_pred = self(train_x)
        training_loss = F.cross_entropy(y_pred, train_y)
        # optional
        self.log('train_loss', training_loss, on_epoch=True, prog_bar=True)
        return training_loss

    def validation_step(self, batch, batch_num):
        # optional
        val_x, val_y = batch
        y_pred = self(val_x)
        val_loss = F.cross_entropy(y_pred, val_y)
        # optional
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self, outputs):
        # optional
        avg_loss = torch.stack(outputs).mean()
        return avg_loss

    def test_step(self, batch, batch_num):
        # optional
        test_x, test_y = batch
        y_pred = self(test_x)
        test_loss = F.cross_entropy(y_pred, test_y)
        # optional
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True)
        return test_loss

    def test_epoch_end(self, outputs):
        # optional
        avg_loss = torch.stack(outputs).mean()
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=0.5)

    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.1302,), (0.3069,))])), 
                                batch_size=32, num_workers=4)

    def val_dataloader(self):
        # optional
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.1302,), (0.3069,))])), 
                                batch_size=32, num_workers=4)

    def test_dataloader(self):
        # optional
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.1302,), (0.3069,))])), 
                                batch_size=32, num_workers=4)
```

`training_step`, `validation_step` , `test_step` : train, validation, testset에서 iteration마다 성능을 평가한다. 

`validation_epoch_end` , `test_epoch_end` : epoch마다 성능을 계산한다.

`train_dataloader`, `val_dataloader` , `test_dataloader` : 데이터셋을 위한 메서드

`configure_optimizer` : 모델을 훈련시킬 때 사용할 optimizer 정의

```python
model = ConvNet()

trainer = L.Trainer(max_epochs=10)    
trainer.fit(model)
```

모델 객체를 인스턴스화 한 뒤, `Trainer` 모듈을 사용해 `trainer` 객체를 정의한다.

하드웨어의 구성에 따라 `L.Trainer` 의 parameter로 `gpus=8` , `tpus=2` 와 같이 추가할 수 있다.

```python
trainer.test()
```

testset 추론

```python
# Start tensorboard.
%reload_ext tensorboard
%tensorboard --logdir lightning_logs/
```

pytorch lightning은 visualization toolkit인 tensorboard와의 간결한 interface를 제공한다. https://www.tensorflow.org/tensorboard

출력 prompt에 나온 링크로 접속하면 tensorboard session을 볼 수 있다.

일반적인 pytorch code로 tensorboard를 사용하려면 https://pytorch.org/docs/stable/tensorboard.html 를 참고하면 된다.

### 그 외

- PyTorch Ignite
- Poutyne