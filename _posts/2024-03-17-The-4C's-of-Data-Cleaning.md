---
title:  "The 4 C’s of Data Cleaning : Correcting, Completing, Creating, and Converting"
date : 2024-03-17 18:00:00 +0900
categories: [ Concepts]
image: "/assets/images/the-4c's-of-data-cleaning.png" 
---  

### The 4 C’s of Data Cleaning : Correcting, Completing, Creating, and Converting

1. **Correcting**
Data Cleaning의 첫 번째 단계는 데이터에서 발견된 오류를 수정하는 것이다. 이는 오타, 잘못된 값, 불일치하는 데이터 등을 수정하여 데이터의 정확성을 보장하는 과정이다. 이를 위해 데이터를 시각적으로 탐색하거나 통계적 기법을 사용하여 이상치를 식별하고, 이를 올바른 값으로 대체하는 작업을 수행한다.
2. **Completing**
Completing 단계는 결측값(missing values)을 다루는 것을 중점적으로 다룬다. 결측값은 데이터에 존재하지 않거나 누락된 값으로, 이를 적절한 방식으로 대체하거나 예측하여 데이터의 완전성을 유지하는 것이 중요하다. 주어진 데이터의 특성에 따라 평균, 중앙값, 최빈값 등의 통계치를 사용하거나 머신 러닝 모델을 활용하여 결측값을 예측하는 방법을 사용할 수 있다.
3. **Creating**
Data Cleaning 과정에서 새로운 변수를 생성하는 것이 필요한 경우가 있다. 이는 기존 변수들을 조합하거나 변형하여 새로운 특성을 만드는 것을 의미한다. 예를 들어, 날짜와 시간 데이터를 결합하여 새로운 시간대 변수를 생성하거나, 범주형 변수를 이진 변수로 변환하는 등의 작업이 이에 해당한다. 이는 데이터의 특성을 더 잘 파악하고 모델의 성능을 향상시키는 데 도움이 된다.
4. **Converting**
마지막으로, Data Cleaning 단계에서는 데이터의 형식이나 척도를 변환하는 작업이 수행된다. 이는 데이터의 일관성을 유지하고 모델 학습에 적합한 형태로 데이터를 준비하는 것을 의미한다. 예를 들어, 범주형 변수를 One-Hot 인코딩하여 숫자 형태로 변환하거나, 연속형 변수를 정규화하여 모든 변수들이 동일한 척도를 가지도록 하는 등의 작업이 여기에 해당합니다.

이렇게 Data Cleaning의 4C인 Correcting, Completing, Creating, Converting 단계를 거쳐 데이터를 준비하면 더 나은 분석과 모델링 결과를 얻을 수 있다.