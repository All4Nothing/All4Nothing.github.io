---
title:  "Optimizer"
date : 2023-11-14 22:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/optimizer.png" 
---  
## Optimizer 최적화 함수

Optimizer(최적화 함수)를 이용한다면, parameter($W$와 $B$) 값을 직접 변경하지 않아도 된다.

```python
W = torch.tensor(1.0, requires_grad = True).float()
B = torch.tensor(1.0, requires_grad = True).float()

num_epochs = 500

lr = 0.01

import torch.optim as optim
optimizer = optim.SGD([W, B], lr=lr)

history = np.zeros((0, 2))

for epoch in range(num_epochs):
	Yp = pred(X)

	loss = MSE(Yp, Y)

	loss.backward()

	optimizer.step()

	optimizer.zero_grad()

if (epoch % 10 == 0):
	item = np.array([epoch, loss.item()])
	history = np.vstack((history, item))
	print(f'epoch = {epoch} loss = {loss:.4f}')
```

optimizer에 `momentum` 옵션을 추가하여 최적화 함수를 tunning 할 수 있다.

```python
optimizer = optim.SGD([W, B], lr=lr, momentum=0.9)
```

### Momentum

SGD 알고리즘을 개선하기 위해 모멘텀(Momentum) 인수를 설정하는 방법이다. SGD가 가장 최근 경사 값만을 파라미터 업데이트에 사용하는 것에 반해, 모멘텀은 과거에 계산했던 경사 값을 기억했다가, 그만큼 파라미터를 일정 비율 감소시켜 파라미터 업데이트에 사용한다.

$s^t_i = \alpha s^{t-1}_i + (1-\alpha )\frac{\delta E^t}{\delta w_i} \\  \Delta w_i^t = -\eta s^t_i$

online learning이나 mini-batch 기법에서 효과적이다. 다만, 추가 메모리가 필요하다($s_i$개수만큼)

```python
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
```

### (Full) Batch Gradient Descent

- 전체 학습 데이터의 gradient를 구해서 파리미터를 업데이트한다.
- 한 스텝당, 전체 데이터셋에 대해 한번의 gradient 계산을 하고, 한번의 파라미터 업데이트를 진행하기에 전체적인 계산 횟수가 적다.
- 다만, 한 스텝당 모든 학습 데이터셋을 사용하므로 학습이 오래걸리고, local optimal 상태에 빠지기 쉽다.

### Stochastic Gradient Descent

- 무작위로 하나의 데이터를 골라 gradient를 계산하고, 파라미터를 업데이트한다.
- 따라서 한 스텝당 걸리는 시간이 짧고, 수렴 속도가 빠르며, global optimal을 찾기 쉽다.
- Dataset이 매우 큰 경우, 매번 전체 학습 데이터를 연산하면 메모리의 사용량이 늘어난다
- Full Batch와는 다르게 확률론적, 불확실성의 개념을 도입하였다

경사에 일정한 확습률을 곱해서 파라미터를 수정해 나간다.

```python
lr = 0.001
W -= lr*W.grad
B -= lr*B.grad
```

PyTorch에서는 다음과 같이 `optim.SGD` 클래스를 사용한다.

```python
import torch.optim as optim
optimizer = optim.SGD([W, B], lr=lr)

for epoch in range(num_epochs):
	...
	optimizer.step()
	optimizer.zero_grad()
	...
```

**local optima** : error의 global 한 minimum지점이 아닌 특정 구간 내에서 minimum한 지점

### mini-Batch Gradient Descent

- Batch GD와 SGD의 절충안으로, 전체 학습 데이터를 minibatch로 나누어, GD를 진행한다

```python
# 1. Pick a mini-batch
# 2. Feed it to Neural Network
# 3. Calculate the mean gradient of the mini-batch (batch GD의 특성 적용)
# 4. Use the mean gradient we calculated in step 3 to update the weights
# 5. Repeat steps 1–4 for the mini-batches we created
```

minibatch GD 장점

- SGD에 비해 local optima에 빠질 위험이 줄어든다
- batch를 나누므로 병렬처리 시 유리하다.
- Batch GD보다 메모리 사용을 절약할 수 있다.

mini-Batch GD와 SGD는 최적화가 진행되면서 fluctuating 곡선을 보인다 → 매번 다른 gradient 평균을 적용하면서 최적화를 진행하기 때문

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06b11ce6-8475-4b57-bdd2-ef3570970ab2/Untitled.png)

### Adam

```python 
optimizer = optim.Adam(net.parameters())
```

*PyTorch Deep Learning Programming Ch10 + PyTorch DeepLearning Project Ch01*