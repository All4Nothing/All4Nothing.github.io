---
title:  "Distributed Training"
metadate: "hide"
date : 2023-12-12 18:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/distributed-training.png" 
---  

## Distributed Training 분산 훈련

GPT3와 같은 large model의 경우, 대체로 수백만에서 수십억 개의 parameter가 있다. 이 많은 parameter를 backpropagation하여 tunning하려면 어마어마한 양의 메모리와 높은 컴퓨터의 성능이 요구되며, 모델을 훈련하는 시간도 오래 걸린다.

Distributed Training은 훈련을 여러 시스템에 그리고 시스템 내 여러 프로세스에 분산시켜 모델 훈련 프로세스의 속도를 높일 수 있다.

**Undistributed Training**

```python
# convnet_undistributed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import time
import argparse

class ConvNet(nn.Module):
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
    
    
def train(args):
    torch.manual_seed(0)
    device = torch.device("cpu")
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1302,), (0.3069,))])),
        batch_size=128, shuffle=True)  
    model = ConvNet()
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    model.train()
    for epoch in range(args.epochs):
        for b_i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss = F.nll_loss(pred_prob, y) # nll is the negative likelihood loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b_i % 10 == 0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i, len(train_dataloader),
                    100. * b_i / len(train_dataloader), loss.item()))
         
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int)
    args = parser.parse_args()
    start = time.time()
    train(args)
    print(f"Finished training in {time.time()-start} secs")
    
if __name__ == '__main__':
    main()
```

`parser` : epoch 수와 같은 hyperparameter를 입력받음

```bash
python convnet_undistributed.py --epochs 5 # python scipt 실행
```

- 컴퓨터 시스템 사양 확인(CPU 코어 개수와 RAM 용량 확인)

```bash
/Volumes/Macintosh\ HD/usr/sbin/system_profiler SPHardwareDataType
```

**Distributed Training**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.multiprocessing as mp
import torch.distributed as dist

import os
import time
import argparse

class ConvNet(nn.Module):
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
     

def train(cpu_num, args):
    rank = args.machine_id * args.num_processes + cpu_num                        
    dist.init_process_group(                                   
    backend='gloo',                                         
    init_method='env://',                                   
    world_size=args.world_size,                              
    rank=rank                                               
    ) 
    torch.manual_seed(0)
    device = torch.device("cpu")
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1302,), (0.3069,))]))  
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
       dataset=train_dataset,
       batch_size=args.batch_size,
       shuffle=False,            
       num_workers=0,
       sampler=train_sampler)
    model = ConvNet()
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    model = nn.parallel.DistributedDataParallel(model)
    model.train()
    for epoch in range(args.epochs):
        for b_i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss = F.nll_loss(pred_prob, y) # nll is the negative likelihood loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b_i % 10 == 0 and cpu_num==0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i, len(train_dataloader),
                    100. * b_i / len(train_dataloader), loss.item()))
         
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-machines', default=1, type=int,)
    parser.add_argument('--num-processes', default=1, type=int)
    parser.add_argument('--machine-id', default=0, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()
    
    args.world_size = args.num_processes * args.num_machines                
    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = '8892'      
    start = time.time()
    mp.spawn(train, nprocs=args.num_processes, args=(args,))
    print(f"Finished training in {time.time()-start} secs")
    
if __name__ == '__main__':
    main()
```

`torch.multiprocessing` : 시스템 내에서 여러 python process를 생성할 수 있게한다.(일반적으로 시스템에 있는 CPU 코어 수만큼의 프로세스를 생성할 수 있다.)

`torch.distributed` : 모델을 훈련시키기 위해 같이 작동하는 다른 시스템 간 통신을 가능하게 한다. 그러면 `Gloo` 와 같은 내장된 pytorch 통신 백엔드 프로그램 중 하나가 시스템 간 통신을 처리한다. 각 시스템 내에서 multi-processing은 여러 process에 걸쳐 훈련 작업을 병렬 처리한다. 

https://pytorch.org/docs/stable/multiprocessing.html

https://pytorch.org/docs/stable/distributed.html

```python
rank = args.machine_id * args.num_processes + cpu_num                        
dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, 
rank=rank) 
torch.manual_seed(0)
device = torch.device("cpu")
```

`rank` : 전체 distributed 시스템에서 process의 순서를 나타내는 ID

- 각각 코어가 4개인 CPU가 있는 2대의 컴퓨터를 사용하면
- 한 시스템(컴퓨터)당 4개의 process를 사용할 수 있다.
- process를 구분하기 위해 두 시스템 ID에 0과 1을 할당하고, 각 시스템의 4개의 process에 0~3의 ID를 할당해 label을 붙일 수 있다.
- $n$번째 시스템의 $k$번째 process의 $rank$

`init_process_group` : process에 대해 각 항목을 설정할 수 있다.

- `backend` : 시스템 간 통신을 위해 사용되는 백엔드(CPU에서 distributed training할 때는 `Gloo`, GPU의 경우 `NCCL` 사용 (https://tutorials.pytorch.kr/intermediate/dist_tuto.html)
- `init_method` : 시스템 전체의 모든 process가 설정한 method를 사용해 시작될 때까지 각 process가 추가 작업을 수행하지 못하도록 차단
- `world_size` : distributed training에 사용되는 전체 process 개수
- `rank` : 시작된 process의 랭크

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, sampler=train_sampler)
model = ConvNet()
optimizer = optim.Adadelta(model.parameters(), lr=0.5)
model = nn.parallel.DistributedDataParallel(model)
model.train()
```

`train_sampler` : train dataset을 `world_size` 수 만큼의 partition으로 분할해 distributed training session의 모든 process가 똑같은 양의 데이터를 가지고 작업하도록 한다. 

- 데이터를 분산할 때 sampler를 사용하므로 dataloader를 인스턴스화할 때 shuffle 옵션은 False로 설정했다.

`DistributedDataParallel` : gradient descent 알고리즘이 distributed 방식으로 작동하게 하는 핵심 API이다. 내부에서는 다음과 같이 작업이 이루어진다.

- distributed 환경에서 생성된 각 process는 고유한 모델 복사본을 얻게 된다.
- process당 각 모델은 자체 optimizer를 유지하고, global iteration과 동기화되는 local optimalization 단계를 거친다.
- 각 distributed training iteration에서 개별 loss 및 gradient가 각 process에서 계산된다.
- 그 후 전체 process에서 구해진 gradient의 평균을 구한다.
- 구해진 평균 gradient는 각 모델 복사본에 global로 backpropagation되어 각 모델의 parameter가 tunning된다.
- global backpropagation 때문에 모든 모델은 각 iteration마다 동일한 paramter를 가진다.(자동 동기화)