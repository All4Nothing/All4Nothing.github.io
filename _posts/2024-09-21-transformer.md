---
title: "Attention Is All You Need"
excerpt: "Transformer Paper Review"

categories:
  - Papers
tags:
  - [Computer Vision, U-Net]
use_math: true

permalink: /papers/transformer/

toc: true
toc_sticky: true

date: 2024-09-21
last_modified_at: 2024-09-21
---
# Attention Is All You Need

- **Link :** https://arxiv.org/abs/1706.03762*

### 💡 Attention의 등장 배경  

기존에는 모델이 자연어를 이해하기 위해 Seq2Seq 구조를 사용했습니다. Seq2Seq 모델은 RNN에서 many-to-many에 해당하는 모델입니다. 그 중 입력 문장을 읽어오는 부분을 ‘인코더(Encoder)’, 출력 문장을 생성하는 부분을 ‘디코더(Decoder)’라고 합니다. 

![image](/assets/images/posts_img/transformer/1.png)

모델이 문장을 읽어올 때, 인코더에서는 문장의 맨 앞 단어부터 순차적으로 읽어와 마지막 hidden state 벡터에 모든 인코딩된 정보를 우겨 넣습니다. 이로 인해, 앞에 나온 단어에 대한 정보는 점차 사라지고, 입력 문장의 길이가 길어지면 Vanishing Gradient와 같은 Long-Term problem이 발생합니다.

이를 해결하기 위해 Attention이라는 개념이 등장하였습니다.

### 💡 Attention   

우리는 문장을 이해할 때, 문장 내의 모든 단어를 집중해서 보지 않습니다. “Attention Is All You Need.”라는 문장이 있으면 우리는 ‘Attention’이라는 단어를 ‘Is’라는 단어보다 더욱 집중해서 보는 것처럼 말이죠.

다시 말해, 예측 단어(Output)을 출력하기 위해, 입력 문장 내에서 Output과 관련이 높은 단어, 즉 중요도가 높은 단어에만 집중(Attention)하자는 컨셉이 바로 Attention입니다.

Attention이란, 디코더가 각 타임 스텝에서 예측 단어를 생성할 때 인코더의 몇 번째 타임 스텝을 더 집중(Attention)해야 하는 지를 점수(Score)형태로 나타내는 것입니다.

디코더의 각 타임 스텝마다 인코더의 hidden state 벡터와의 유사도를 계산하여, 인코더의 몇 번째 hidden state 벡터가 더 필요한지(중요한지)를 고려할 수 있게 됩니다.

### 💡 Seq2Seq with Attention  

기존 Seq2Seq 모델에 Attention 기법을 적용하면 어떻게 될까요?

기존 RNN 기반 Seq2Seq 구조의 경우, ‘이전 타임 스텝의 hidden state 벡터(Output)’와 ‘현재 타임 스텝의 디코더 입력값’을 통해 ‘디코더의 hidden state’를 구했습니다.

Attention 구조가 추가된 Seq2Seq에서는, 현재 타임 스텝의 디코더 hidden state와 각각의 인코더 타임 스텝의 hidden state 벡터들을 내적(Dot-product)하여 Attention score를 구합니다.

구해진 Attention score를 인코더 hidden state 벡터들의 가중치(weight)로 사용하여, 가중 평균하여 Attention 벡터, 즉 하나의 output 벡터를 구해줄 수 있습니다. 이렇게 구해진 Attention 벡터와 디코더의 마지막 타임 스텝의 hidden state 벡터를 concat하여 마지막 output layer의 입력으로 넣어줍니다.

![image](/assets/images/posts_img/transformer/2.png)


출처: 위키독스 딥러닝을 이용한 자연어 처리 입문

### 💡 Transformer  

Transformer는 오로지 Attention Mechanism에만 의존한 simple network 구조를 가집니다.

Attention mechanism은 입력 문장(input sequence)과 출력 문장(output sequence)의 거리에 상관없이 의존성(dependency)를 모델링할 수 있지만, 앞서 언급한 모델처럼 대부분의 경우 recurrent network와 함께 사용되고 있었습니다.

Recurrent 모델은 구조상 병렬 처리를 할 수 없고, 메모리 제약에 따라 문장(sequence)의 길이가 길어지면 그에 따른 학습에 문제가 생깁니다.

따라서, Transformer는 recurrence없이 오로지 attention mechanism에만 의존하여 입력과 출력 간의 전역 의존성(Global Dependency)를 모델링 할 수 있는 모델입니다.

더 이상 RNN이나 CNN 모듈은 필요하지 않고, Attention Mechanism만 있으면 되기에 논문의 제목 또한 Attention Is All You Need라는 점을 확인할 수 있습니다.

### 💡 Query, Key, Value로 Attention Vector 구하기  

우리가 유사도를 구하고자 하는 vector를 Query, 그 Query와의 유사도를 계산할 다른 벡터들을 Key라 합니다.

이렇게 구한 유사도 점수에 Softmax를 적용하여 attention score를 구하고, 각 벡터들의 값인 Value와(즉, key의 개수와 value의 개수는 동일) 가중 평균하여 Attention 벡터를 구합니다. 이 Attention 벡터가 Query의 hidden state값으로 들어가게 됩니다.

Query, Key, Value는 입력 단어의 Embedding 값(X)에 각각의 가중치 행렬을 곱하여 구할 수 있습니다.

- $XW^Q=Q,\ XW^K=K,\ XW^V=V$
    
    ![image](/assets/images/posts_img/transformer/3.png)
    
    출처: 위키독스 딥러닝을 이용한 자연어 처리 입문
    

단일 query q에 대해, key들의 행렬인 K와 value들의 행렬인 V가 있을 때 Attention 벡터를 구하는 과정을 수식으로 표현하면 다음과 같습니다.

$A(q,K,V)= \sum_i \frac{\exp(q\cdot k_i)}{\sum_j\exp(q\cdot k_j)}v_i$

사실 이렇게 벡터 단위로 각 단어에 대한 Attention 벡터를 구하는 대신, 행렬 단위로 계산하여 Attention 행렬을 구할 수 있습니다.

기존에는 각 단어에 대한 벡터 계산을 통해 Attention score를 구했다면, 행렬을 이용하여 문장 내 모든 단어에 대한 행렬 계산을 통해 Attention 행렬을 구할 수 있습니다.

![image](/assets/images/posts_img/transformer/4.png)

출처: 위키독스 딥러닝을 이용한 자연어 처리 입문

![image](/assets/images/posts_img/transformer/5.png)

출처: 위키독스 딥러닝을 이용한 자연어 처리 입문

$A(Q,K,V) = softmax(QK^T)V$

![image](/assets/images/posts_img/transformer/6.png)

이러한 행렬 계산 방식을 통해(논문에서는 *‘highly optimized matrix multiplication code’*라고 표현함) 기존 RNN 계열의 모델보다 속도와 공간 측면에서 이점을 갖게 됩니다.

### 💡 Scaled Dot-Product Attention  

Attention score를 계산 시 query와 key의 dimension에 따라 내적의 분산값이 좌지우지 될 수 있고 이에 따라 Softmax의 분포에 영향을 줄 수 있습니다. 이를 보정해주기 위해 표준편차로 나눠주는 과정을 통해 분산을 1로 유지할 수 있습니다. Q와 K가 평균이 0이고 분산이 1인 vector로 이루어져 있다면, 통계적으로 계산 했을때 $Q\cdot K$의 분산값과 $d_k$의 값이 동일합니다. 따라서, Q와 K의 **Dot-Product** 값을 key의 dimension인 $d_k$로 나눠(**Scaled**) 최종적으로 **Attention** 벡터값을 구할 수 있습니다.

### 💡 Multi-Head Attention  

Multi-Head Attention을 활용하면 동시에 여러 버전의 Attention을 진행할 수 있습니다. 한 헤드(head)는 한 종류의 가중치 행렬($W_i^Q, W_i^K, W_i^V$)을 통해 Q, K, V를 구하고 최종적으로 Attention 벡터를 계산합니다. 만약 이 헤드가 여러 개 있다면? 우리는 여러 종류의 가중치 행렬($head_0=Attention(QW_0^Q,KW_0^K,VW_0^V), \ head_1=Attention(QW_1^Q,KW_1^K,VW_1^V) ...$)을 이용하여 여러 버전의 Attention 벡터를 구할 수 있습니다. 이렇게 각 헤드별로 얻어진 Attention 벡터를 concat하여 전체 결과 벡터를 얻을 수 있습니다. 이를 통해, 문장을 여러 관점에서 바라볼 수 있게 됩니다.

### 💡 3가지 Attention  

![image](/assets/images/posts_img/transformer/7.png)

Transformer에서는 3가지의 Multi-Head Attention을 사용합니다.

- Encoder Self-Attention: Self-Attention은 자기 자신에게 attention을 수행한다는 의미로, 쉽게 말해 인코더로 들어온 입력 문장의 모든 벡터들에 대해 각각 Attention 벡터를 구한다는 뜻입니다. 따라서, 이때 Q, K, V는 입력 문장의 모든 벡터들이 해당됩니다. 이렇게 self-attention을 통해 입력 문장 내의 단어들끼리의 유사도를 구할 수 있습니다.
    
    ![image](/assets/images/posts_img/transformer/8.png)

    

- Masked Decoder Self-Attention: RNN은 구조적으로 다음 단어를 예측할 때, 이전까지 입력된 단어들만을 참고할 수 있었습니다. 하지만 Transformer는 문장 행렬을 입력으로 받기에 다음 단어를 예측할 때, 그 뒤에 나오는 단어들까지도 참고할 수 있습니다. 이를 방지하기 위해, Masking을 하여 Attention score matrix에서 자기 자신(디코더로 들어오는 embedding)과 그 이전에 나온 단어들만 참고할 수 있게 합니다.
    
    ![image](/assets/images/posts_img/transformer/9.png)

    
    ![image](/assets/images/posts_img/transformer/10.png)

    

- Encoder-Decoder Attention: 이번에는 Self-Attention이 아닌, 디코더에서의 Query에 대해 인코더의 마지막 층에서 나Key와 Value를 이용해 Attention을 진행합니다.
    
    ![image](/assets/images/posts_img/transformer/11.png)

    

### 💡 Residual Connection & Layer Normalization **(Add&Norm)**  

![image](/assets/images/posts_img/transformer/12.png)


block(모듈)을 보면 3개의 벡터가 각각 query, key, value로 들어가서 Multi-Head Attention을 통해 계산되고, 이렇게 Attention을 거친 임베딩 벡터와 원래의 임베딩 벡터를 더해주는 것(Add)을 residual connection이라 부릅니다. Residual connection을 통해 레이어가 깊어질수록 gradient가 점점 커지거나 작아지는 문제를 해결할 수 있습니다.

그 후 Layer Normalization을 통해 각 입력값들의 Feature들에 대한 평균과 분산을 구해 batch에 있는 각 입력값들을 정규화 해줍니다.

### 💡 Positional Encoding  

RNN은 모델의 구조상 자연스럽게 단어들의 순서에 대한 정보를 학습하지만, Attention 모델은 구조상 순서 정보를 학습하지 않습니다. 따라서 각 단어의 포지션마다 그 포지션을 나타내는 정보를 추가해주는 것을 Positional Encoding이라 합니다.  Positional encoding은 다음과 같은 사인파 함수를 이용하고, 임베딩 벡터 내의 차원 인덱스에 따라 sin함수와 cos함수를 이용하여 계산합니다.

![image](/assets/images/posts_img/transformer/13.png)


### 💡 Transformer 모델의 장점  

이러한 Transformer 모델은 

- Parallelization: 병렬화를 통해 계산 효율성을 높이고 학습속도를 개선할 수 있었고
- Long-range Dependencies: 모든 위치에서 다른 위치까지의 관계(Long-Term Dependency)를 쉽게 학습할 수 있었으며
- Interpretable: 또한 Attention score를 시각화하여 각 요소들 간의 관계를 시각화하여 볼 수 있습니다.

![image](/assets/images/posts_img/transformer/14.png)
