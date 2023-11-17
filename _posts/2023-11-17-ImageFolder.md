---
title:  "ImageFolder"
metadate: "hide"
date : 2023-11-17 18:00:00 +0900
categories: [ ML/DL ]
image: "/assets/images/ImageFolder.png" 
---  
## ImageFolder

내장 함수로 호출 가능한 공개 데이터셋이 아닌, 직접 수집한 JPEG 형식과 같은 이미지 데이터를 사용하는 방법에 대해 알아본다.

### Data Download

github 레포지토리 등에 이미지 파일을 업로드 해 둔 경우 다음과 같이 다운이 가능하다.

```python
w = !wget -nc https://github.com/All4Nothing/dataset/hymenoptera.zip
```

그렇지 않은 경우, 다음 command를 통해 직접 파일을 업로드 할 수 있다.

```python
from google.colab import files
files.upload()
```

### Unzip

```python
w = !unzip -o hymenoptera_data.zip
```

`-o` 옵션은 unzip을 두 번 이상 실행할 경우 파일을 덮어쓰게 하기 위함이다.

### tree command install

```python
!pip install torchviz | tail -n 1
!pip install torchinfo | tail -n 1 
w = !apt install tree
```

실행 결과를 `w`에 대입한 이유는 OS command의 출력을 숨기기 위함이다.

다음과 같이 압축 해제한 파일의 tree 구조를 출력할 수 있다.

```python
!tree hymenoptera_data
```

### ImageFolder

ImageFolder를 사용해서 데이터셋을 정의할 수 있다.

- directory 및 class 정의

```python
# base dir
data_dir = 'hymenoptera_data'

# train and test dir
import os
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

# class list
classes = ['ants', 'bees']
```

- dataset 정의

```python
# train dataset
train_data = datasets.ImageFolder(train_dir, transform = train_transform)

# test dataset
test_data = datasets.ImageFolder(test_dir, transform = test_transform)
```

- dataloader 정의

```python
batch_size = 10

# train dataloader
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

# test dataloader
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)
```