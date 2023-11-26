---
title:  "Data Augmentation"
metadate: "hide"
date : 2023-11-14 19:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/data-augmentation.png" 
---  
## Data Augmentation 데이터 증강

데이터 증강은 학습 전 입력 데이터를 인위적으로 가공해서 학습 데이터의 다양성을 증가시키는 방법이다. 모델의 관점에서는 학습을 반복할 때마다 다른 패턴의 데이터가 들어오기 때문에, overfitting을 예방할 수 있다.

PyTorch에서는 `Transforms` 기능을 통해 데이터 증강을 할 수 있다.

PyTorch 사용 가능 기능 목록
- RandomApply
- RandomChoice
- RandomCrop
- RandomResizedCrop
- ColorJitter
- RandomGrayscale
- RandomHorizontalFlip
- RnadomVerticalFlip
- RandomAffine
- RandomPerspective
- RandomRotation
- RandomErasing

```python
transform_train = transforms.Compose([
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.ToTensor()
	transforms.Normalize(0.5, 0.5),
	transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])
```

