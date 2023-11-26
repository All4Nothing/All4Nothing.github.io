---
title:  "Lambda Expression"
metadate: "hide"
date : 2023-11-03 20:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/lambda-expression.png" 
---  

## Lambda Expression 람다 표현식

```python
# 기존 방식
def f(x):
	return x.view(-1)

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(0.5, 0.5),
	transforms.Lambda(f),
])

# lambda expression
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(0.5, 0.5),
	transforms.Lambda(lambda x: x.view(-1)),
])
```

Lambda 클래스를 호출함과 동시에 함수의 정의가 가능하다. `f`라는 함수는 여기서만 사용하므로 `f`라는 이름을 정의하는 것도 사실은 필요 없으며, 람다 표현식을 사용하는 경우는 함수에 이름을 붙이고 있지 않다. 이와 같은 사용법을 ‘**무명 함수(Anonymous function)**’라고 부르기도 한다.