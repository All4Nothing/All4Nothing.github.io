---
title:  "Text Generation with PyTorch"
date : 2023-11-19 18:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/text-generation-with-pytorch.png" 
---  
## Text Generation with PyTorch 파이토치로 텍스트 생성기 만들기

Transfomer 또는 GPT-2와 같은 모델을 기반으로 text generator를 만들 수 있다. Language 모델을 학습시키고 초기 단어의 sequence를 cue로 제공하면, iteration을 돌면서 input sequence에 현재 iteration에서 예측된 단어를 덧붙여가며 sequence를 확장 시킬 수 있다. 만들어진 sequence는 다음 iteration에서 모델의 input이 된다. Text generator의 성능은 기반이 되는 language 모델의 성능에 따라 달라진다.

```python
ln = 10
sntc = 'It will _'
sntc_split = sntc.split()
with torch.no_grad():
    for i in range(ln):
        sntc = ' '.join(sntc_split)
        txt_ds = TEXT.numericalize([sntc_split])
        num_b = txt_ds.size(0)
        txt_ds = txt_ds.narrow(0, 0, num_b)
        txt_ds = txt_ds.view(1, -1).t().contiguous().to(device)
        ev_X, _ = return_batch(txt_ds, i+1)
        op = transformer_cached(ev_X)
        op_flat = op.view(-1, num_tokens)
        res = TEXT.vocab.itos[op_flat.argmax(1)[0]]
        sntc_split.insert(-1, res)
print(sntc[:-2])
```
*It will be used to the song song , and the*

### Text Generation Technique

**Greedy**

Greedy 탐색은 모델이 현재 iteration에서 최대 확률을 선택한다. 

```python
ip_ids = tkz.encode(cue, return_tensors='pt')
op_greedy = mdl.generate(ip_ids, max_length = ln)
seq = tkz.decode(op_greedy[0], skip_special_tokens=True)
print(seq)
```

**Beam**

Beam Search는 다음 단어 확률이 아닌 전체 예측 시퀸스의 확률을 기반으로 다른 후보 시퀸스의 목록을 유지하는 방식이다. 빔 크기를 설정해 후보 시퀸스의 개수를 정할 수 있다. 예를 들어 빔 크기가 3인 빔 서치에서는 iteration 마다 가장 확률이 높은 3개의 후보 시퀸스가 유지된다. 

```python
op_beam = mdl.generate(
    ip_ids, 
    max_length=5, 
    num_beams=3, 
    num_return_sequences=3, 
)

for op_beam_cur in op_beam:
    print(tkz.decode(op_beam_cur, skip_special_tokens=True))
```

![greedy-beam](https://heidloff.net/assets/img/2023/08/greedy-beam.jpeg)

### Top-k & Top-p sampling

가장 확률이 높은 단어를 선택하는 대신, 확률에 기반에 단어를 무작위로 샘플링 한다. 예를 들어, 다음 단어로 올 확률이 ‘woman’이 0.5, ‘house’는 0.3,  ‘guy’는 0.2이라고 할 때, 확률이 가장 높은 ‘woman’을 선택하는 것이 아닌 0.5의 확률로 ‘woman’이, ‘0.3’의 확률로 ‘house’가 샘플링 되는 방식이다. 이를 통해 greedy와 beam search와는 달리 다양한 조합의 sequence를 생성할 수 있다.

Top-k sampling에서는 다음 후보가 될 단어의 개수인 매개변수 $k$를 정의하고, 상위 $k$개의 단어에서 확률을 정규화한다. $k$가 2라면, ‘guy’는 제외되고, ‘woman’과 ‘house’의 확률이 0.625, 0.375로 정의 되는 식이다. 

```python
for i in range(3):
    torch.manual_seed(i)
    op = mdl.generate(
        ip_ids, 
        do_sample=True, 
        max_length=5, 
        top_k=2
    )

    seq = tkz.decode(op[0], skip_special_tokens=True)
    print(seq)
```

top-p 샘플링에서는 누적 확률 threshold로 $p$를 정의하여, 누적 확률이 p가 될 때가지 나온 단어를 모두 유지한다. 예를 들어 $p$가 0.5와 0.7 사이라면 ‘woman’만이(누적 확률 0.5) , $p$가 0.8과 0.9 사이라면 ‘woman’과 ‘house’가 (누적 확률 0.8)이 유지된다. $p$가 1이면 모든 단어가 유지된다.

확률이 고르게 분포되어 있을 경우, top-k보단 top-p sampling이 더 유리할 수 있다.ㅌ

```python
for i in range(3):
    torch.manual_seed(i)
    op = mdl.generate(
        ip_ids, 
        do_sample=True, 
        max_length=5, 
        top_p=0.75, 
        top_k=0
    )

    seq = tkz.decode(op[0], skip_special_tokens=True)
    print(seq)
```