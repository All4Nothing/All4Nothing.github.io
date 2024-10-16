---
title: "NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE"
excerpt: "NEURAL MACHINE TRANSLATION Paper Review"

categories:
  - Papers
tags:
  - [NLP]
use_math: true

permalink: /papers/neural-machine-translation/

toc: true
toc_sticky: true

date: 2024-06-03
last_modified_at: 2024-06-03
---
# NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE

- ***Link :** https://arxiv.org/abs/1409.0473*  

기존에 제안된 대부분의 Neural Machine Translation 모델은 encoder-decoders구조임. 이 encoder-decoder 접근 방식의 잠재적인 이슈는 neural network가 source sentence의 모든 필요한 정보를 fixed-length vector로 압축할 수 있어야 한다는 것임. 이것은 NN이 긴 문장, 특히 training corpus(말뭉치; NLP에서 모델을 학습하기 위한 데이터)에 있는 문장보다 긴 문장을 다룰때 문제가 생김.  
이러한 문제를 해결하기 위해, 모델이 단어를 번역할 때, source sentence의 부분집합 중 가장 적절한 정보들이 모여있는 집합을 찾아 사용함. 좀 더 기술적으로 말하자면, translation을 decoding하는 동안 input sentence를 vector들의 sequence로 encoding하고, 이 벡터들의 부분집합을 적절히 선택해 사용함.  
이런 접근 방식은 기존 basic encoder-decoder 접근 방식보다 능가하는 성능을 보였고, 특히 sentence가 길어질수록 그 성능이 더욱 돋보임