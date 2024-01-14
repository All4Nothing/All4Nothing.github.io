---
title:  "PyTorch GPU"
date : 2023-10-31 20:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/pytorch-gpu.png" 
---  

## PyTorch GPU 사용규칙

PyTorch에서 GPU를 사용할 경우 다음과 같은 규칙을 지켜야 한다.

1. 텐서 변수는 데이터가 CPU와 GPU 중 어디에 속해 있는 지를 속성으로 갖는다.
2. CPU와 GPU 사이에서 데이터는 to 함수로 전송한다.
3. 두 개의 변수가 모두 GPU에 올라가 있는 경우, 연산은 GPU로 수행한다.
4. 두 변수 중 한쪽이 CPU, 다른 한쪽이 GPU에 올라가 있는 경우, 연산은 에러를 발생시킨다.

```python
x_np = np.arange(-2.0, 2.1, 0.25)
y_np = np.arange(-1.0, 3.1, 0.25)
x = torch.tensor(x_np).float()
y = torch.tensor(y_np).float()

z = x * y
print(z)

x = x.to(device)
print('x: ', x.device)
print('y: ', y.device)
```

```
Tensor([2.0000, 1.3125, 0.7500, 0.3125, -0.0000, -0.1875, -0.2500, -0.1875, 0.0000, 0.3125, 0.7500, 1.3125, 2.0000, 2.8125, 3.7500, 4.8125, 6.0000])
x: cuda:0
y: cpu
```

```python
z = x * y
```

```
Runtime Error Traceback(most recent call last)
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

서로 다른 디바이스에 위치한 데이터 간 연산은 이처럼 런타임 에러를 발생시키는 것을 확인할 수 있다. 이것이 바로 규칙4에 해당하는 사실이다.

```python
y = y.to(device)
z = x * y
print(z)
```

```
Tensor([2.0000, 1.3125, 0.7500, 0.3125, -0.0000, -0.1875, -0.2500, -0.1875, 0.0000, 0.3125, 0.7500, 1.3125,
2.0000, 2.8125, 3.7500, 4.8125, 6.0000], device='cuda:0')
```

> 해당 포스팅은 '차근차근 실습하며 배우는 파이토치 딥러닝 프로그래밍'의 Ch08을 참고하여 작성되었습니다.