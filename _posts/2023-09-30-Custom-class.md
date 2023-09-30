---
title:  "Custom Class"
metadate: "hide"
date : 2023-09-30 16:30:00 +0900
categories: [ ML/DL ]
image: "/assets/images/custom-class.jpeg" 
visit: "https://github.com/All4Nothing"
---

## Custom Class  

### 객체 지향(Object-Oriented)
Python과 객체 지향(Object-Oriented)과의 관계는, 크게 세 가지 수준으로 나눌 수 있다.

1. 객체(object)라는 개념과 무관하게 프로그램을 작성하는 수준
2. 객체는 사용하지만 기존 class의 instance를 생성해서 사용하는 수준 (ex. 사이킷런(scikit-learn)을 사용한 머신러닝 모델 구축
3. 가장 높은 수준인 사용자의 프로그램 안에 독자적인 class를 정의하는 패턴. 독자적인 class의 정의는 'custom class 정의'라고도 한다.

객체 지향에서 class는 ‘틀’에 해당하는 개념이고, instance는 이 ‘틀’로부터 생성된 개별적인 실체이다. class는 ‘property’라고 하는 class 안의 변수를 갖는다. 또한, 함수나 method도 존재한다. 
쿠키틀로 쿠키를 찍어낸다고 이해하면 쉽다.  

```python
# Point 클래스 정의
class Point:
    # 인스턴스 생성 시에 두 개의 인수 x와 y를 가짐
    def __init__(self, x, y):
        # 인스턴스 속성 x에 첫 번째 인수를 할당
        self.x = x
        # 인스턴스 속성 y에 두 번째 인수를 할당
        self.y = y

    # draw 함수 정의
    def draw(self):
        # (x, y)에 점을 그림
        plt.plot(self.x, self.y, marker='o', markersize=10, c='k')
```

`__init__` 함수는 class로부터 instance를 생성할 때, 초기화 처리를 위해 반드시 호출되는 함수다. `__init__` 함수의 첫 번째 인수 `self` 는 class로부터 instance를 생성할 때, instance 자신을 가리킨다. `self.x = x` 와 같이 instance의 property(`self.x`)에 instance의 parameter인 `x` 를 대입할 수 있다. 

```python
# Point의 자식 클래스 Circle 정의 1

class Circle1(Point):
    # Circle은 인스턴스 생성 시에 인수 x, y, r을 가짐
    def __init__(self, x, y, r):
        # x와 y는 부모 클래스의 속성으로 설정
        super().__init__(x, y)
        # r은 Circle의 속성으로 설정
        self.r = r
```

`class Circle1(Point)` 와 같이 `Circle1` class를 `Point` class의 자식 class로 정의할 수 있다.  `super().__init__(x,y)` 는 부모 class인 `Point` class의 `__init__` 함수를 호출한다.

```python
# Point의 자식 클래스 Circle의 정의 2

class Circle2(Point):
    # Circle은 인스턴스 생성 시에 인수x, y, r을 가짐
    def __init__(self, x, y, r):
        # x와 y는 부모 클래스의 속성으로 설정
        super().__init__(x, y)
        # r은 Circle의 속성으로 설정
        self.r = r
     
    # draw 함수는 자식 클래스만 따로 원을 그림
    def draw(self):
        # 원 그리기
        c = patches.Circle(xy=(self.x, self.y), radius=self.r, fc='b', ec='k')
        ax.add_patch(c)
```

`Circle2` class 내부에 부모 class인 `Point` class의 `draw` 함수와 같은 이름의 함수를 재정의할 수 있다. 이 경우 `Circle2` class로 만든 instance에서  `draw` 함수를 호출할 경우 부모 class인 `Point`

의 `draw` 함수가 호출되는 것이 아닌, `Circle2` 의 `draw` 함수가 호출된다. 

이처럼 자식 class 내부에 부모 class와 같은 이름의 함수를 재정의하는 것을 `override` 라고 부른다. 

```python
# Point의 자식 클래스 Circle의 정의 3

class Circle3(Point):
    # Circle은 인스턴스 생성 시에 인수 x, y, r을 가짐
    def __init__(self, x, y, r):
        # x와 y는 부모 클래스의 속성으로 설정
        super().__init__(x, y)
        # r은 Circle의 속성으로 설정
        self.r = r
     
    # Circle의 draw 함수는 부모의 함수를 호출한 다음, 원 그리기를 독자적으로 수행함
    def draw(self):
        # 부모 클래스의 draw 함수 호출
        super().draw()
        
        # 원 그리기
        c = patches.Circle(xy=(self.x, self.y), radius=self.r, fc='b', ec='k')
        ax.add_patch(c)
```

`super().draw()`와 같이 자식 class에서 부모 class의 함수를 호출할 수도 있다.