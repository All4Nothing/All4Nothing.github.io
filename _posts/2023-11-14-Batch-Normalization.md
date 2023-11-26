---
title:  "Batch Normalization"
metadate: "hide"
date : 2023-11-14 18:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/batch-normalization.png" 
---  

## Batch Normalization 배치 정규화

전 레이어 함수의 출력을 미니 배치 단위로 정규화 처리를 거친 다음, 이어지는 레이어 함수로 입력하면, 학습 효율이 향상됨과 동시에 overfitting도 예방할 수 있다. 이 알고리즘을 배치 정규화(Batch Normalization)라고 한다.

BN 레이어 함수는 다음과 같은 특징이 있다.

- 합성곱 연산 중에는 `nn.BatchNorm2d` 를, 선형 함수 바로 뒤에는 `nn.BatchNorm1d` 를 사용한다.
- 인스턴스 생성 시 `nn.BatchNorm2d` 를 사용할 때는 입력 데이터의 채널 수, `nn.BatchNorm1d` 를 사용할 때는 입력 데이터의 차원 수를 파라미터로 넣어준다.
- BN 레이어 함수는 학습 대상 파라미터인 `weight` 와 `bias` 를 가지고 있다. 따라서, 각 BN 레이어마다 별도의 인스턴스를 생성해 사용해야 한다.
- 훈련 페이즈와 예측 페이즈가 존재한다.

배치 정규화에서는 다음과 같이 계산이 된다.

```python
torch.manual_seed(123)
inputs = torch.randn(1, 1, 10)
print(inputs)
```

tensor([[[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969,  0.2093, -0.9724, -0.7550,
          0.3239, -0.1085]]])

```python
i_mean = inputs.mean()
i_var = inputs.var(unbiased=True)
i_std = inputs.std(unbiased=False)
print(i_mean, i_std, i_var)
```

tensor(-0.3101) tensor(0.4867) tensor(0.2632)

```python
bn= nn.BatchNorm1d(1)
print(bn.running_mean)
print(bn.running_var)
print(bn.weight.data)
print(bn.bias.data)
```

tensor([0.])
tensor([1.])
tensor([1.])
tensor([0.])

`running_mean` 과 `running_var` 는 예측 페이즈일 때 사용하는 변수로, BN 함수를 호출할 때마다 값이 자동으로 바뀌게 된다.

`weight` 와 `bias`는 학습 대상인 파라미터이다. 학습이 진행될 때마다 최적의 값을 찾아 변해간다.

훈련 페이즈에서는 다음과 같이 계산된다.

$y = \frac{x - E[x]]}{\sqrt{Var[x] + \epsilon }} * r + \beta$

```python
# train
xt = (inputs - i_mean) / i_std * bn.weight + bn.bias
```

예측 페이즈에서는 다음과 같이 계산된다.

```python
# eval
xp = (inputs - bn.running_mean) / torch.sqrt(bn.running_var)
```

running_mean과 running_var는 다음과 같이 업데이트 된다.

$\widehat{x}_{new} = (1-momentum) \times \widehat{x} + momentum \times \widehat{x}_t$

```python
mean0 = 0
var0 = 1
momentum = bn.momentum

# epoch 1
mean1 = (1 - momentum) * mean0 + momentum * i_mean
var1 = (1 - momentum) * var0 + momentum * i_var

# epoch 2
mean2 = (1 - momentum) * mean1 + momentum * i_mean
var2 = (1 - momentum) * var1 + momentum * i_var
```

> 해당 포스팅은 '차근차근 실습하며 배우는 파이토치 딥러닝 프로그래밍'의 Ch10을 참고하여 작성되었습니다.