---
title:  "Loss vs Loss Function vs Criterion"
date : 2023-10-21 17:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/loss-vs-lossfunction-vs-criterion.png" 
visit: "https://github.com/All4Nothing"
---  
## Loss vs Loss Function vs Criterion
손실 함수를 다룰때 사용하는 각 명칭의 미세한 차이점을 알아보자. 
### Loss
`loss`는 경험상 주로 `loss value`를 표현하기 위해 사용한다.  
```python
loss = criterion(outputs, labels)
```
### Criterion
`criterion`은 최소화 또는 최대화하길 원하는 `objective function`으로, 전형적으로 `callable`한 function이나 `nn.Module instance`를 말한다.  
```python
criterion = nn.CrossEntropyLoss()
```  
### Loss Function
`loss function`은 명칭에서 분명하게 밝히듯, 최소화하기를 원하는 `objective function`이다.