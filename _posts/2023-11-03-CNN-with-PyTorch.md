---
title:  "CNN with PyTorch"
metadate: "hide"
date : 2023-11-03 18:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/cnn-with-pytorch.png" 
---  

## CNN with PyTorch 파이토치로 CNN 모델 만들기
### Convolution 합성곱 처리

커널은 주로 홀수(3x3, 5x5)로 하는 것이 일반적이다. 
커널은 입력 채널의 분량만큼 있기 때문에, 4계 텐서 구조를 갖는다.
이 4계 텐서인 커널 배열이 신경망에서 '파라미터'에 해당하고, 이 텐서를 이루는 값이 학습 대상이 된다.
합성곱 처리는 커널의 내용에 따라 특정한 기울기를 가진 직선이 강조되는 등, 도형의 특징량을 추출하는 데 탁월하다. 그리고 학습은 커널이 위치를 이동하면서 이뤄지므로, **위치의 이동과 관련이 없는 특징량**을 검출할 수 있게 된다.

### Pooling 풀링 처리

합성곱 처리는, 화소를 한 개씩 옮겨가며 처리하는 과정이 대부분인 데에 반해, 풀링 처리 는 중첩되는 영역이 없게끔 옮겨가며 처리하는 것이 일반적이다. 사각형 영역의 사이즈는 대부분 2x2를 선호하므로, **가로세로가 모두 원본 이미지의 절반의 화소 수를 갖는 새로 운 이미지가 완성**된다.
풀링 처리를 통해 이미지를 축소하는 것과 동일한 효과를 얻을 수 있다. 따라서, **물체의 크기와 관련이 없는 보편적인 특징량을 추출하는 것에 특화되어 있다.**

- '합성곱 함수(conv)'는 학습 대상이므로 함수의 내부에 파라미터를 갖는다. 이에 반해, '풀링 함수'는 단지 연산에 불과하므로 파라미터가 필요하지 않다. 
마찬가지로, 선형함수(l)는 파라미터를 갖지만, ReLU 함수는 파라미터를 갖지 않는다.

```python
# layer
conv1 = nn.Conv2d(3, 32, 3)
relu = nn.ReLU(inplace = True)
conv2 = nn.Conv2d(32, 32, 3)
maxpool = nn.MaxPool2d((2,2))

print(convl)
print(convl.weight.shape)
print(conv1.bias.shape)
print(conv2.weight.shape)
print(conv2.bias.shape)
```

```
Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
torch.Size([32, 3, 3, 3])
torch.Size([32])
torch.Size([32, 32, 3, 3])
torch.Size([32])
```

`nn.Conv2d(입력 채널 수, 출력 채널 수, 커널 사이즈)'`

`nn.MaxPool2d(커널 사이즈)`
ReLU는 파라미터를 갖지 않는 단순한 함수이므로, 같은 함수를 반복적으로 사용하기 위해 한 번만 정의한다.
Conv 함수는 파라미터를 갖고 있어서 각 레이어마다 개별적으로 정의해줘야 한다.

```python
inputs = torch.randn(100, 3, 32, 32)
print(inputs.shape)

# CNN simulation
x1 = conv1 (inputs)
x2 = relu(x1)
x3 = conv2(x2)
x4 = relu(x3)
x5 = maxpool(x4)

# print shape
print(inputs.shape)
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)
print(x5.shape)
```

```
torch.Size([100, 3, 32, 32]) # 데이터 건수, 채널 수, 화소 수, 화소 수
torch.Size([100, 3, 32, 32])
torh.Size([100, 32, 30, 30])
torh.Size([100, 32, 30, 30])
torh.Size([100, 32, 28, 28])
torh.Size([100, 32, 28, 28])
torh.Size([100, 32, 14, 14])
```

합성곱 함수를 한 번 거칠 때마다, '32→30', '30→28'과 같이 2만큼의 간격으로 줄어들고 있다. 합성곱 처리를 한 번 거칠 때마다 '합성곱 처리 행렬의 사이즈-1'만큼 출력 데이터의 화소 수가 줄어든다.
풀링 처리에서는 가로와 세로 모두 사이즈가 절반인 14가 된다.
ReLU 함수에서는 사이즈의 변화가 일어나지 않는다.

### nn.Sequential

`nn.Sequential`이라는 class는 PyTorch에서 'container’라고 불리는 class 중 하나다.

```python
features = nn.Sequential(
	convl,
	relu,
	conv2, 
	relu,
	maxpool
)

outputs = features(inputs)
print(outputs.shape)
```

```
torch.Size([100, 32, 14, 14])
```

### nn.Flatten

`nn.Flatten` 함수는 합성곱 처리와 풀링 처리가 이뤄질 때 사용된 3계 텐서의 형태의 데이터를, 선형 함수(nn.Linear)에서 사용할 수 있도록 1계 텐서의 형태로 변환해 준다.

```python
flatten = nn.Flatten()
outputs2 = flatten(outputs)
print(outputs.shape)
print(outputs2.shape)

```

```python
torch.Size([100, 32, 14, 14])
torch.Size([100, 6272]) # 32 x 14 x 14 = 6272
```

https://c231n.github.io/convolutional-networks/#overview 참고

```python
# 손실 계산 함수
def eval_loss (loader, device, net, criterion):
	# dataloader에서 처음 한 개 세트를 가져옴
	for images, labels in loader:
		break

	inputs = images.to(device)
	labels = labels.to(device)

	outputs = net(inputs)
	
	loss = criterion(outputs, labels)
	
	return loss
```

```python
from tqdm.notebook import tqdm
# 학습 함수
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):

    # tqdm 라이브러리 임포트
    from tqdm.notebook import tqdm

    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs+base_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # 훈련 페이즈
        net.train()
        count = 0

        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 경사 초기화
            optimizer.zero_grad()

            # 예측 계산
            outputs = net(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 경사 계산
            loss.backward()

            # 파라미터 수정
            optimizer.step()

            # 예측 라벨 산출
            predicted = torch.max(outputs, 1)[1]

            # 정답 건수 산출
            train_acc += (predicted == labels).sum().item()

            # 손실과 정확도 계산
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        # 예측 페이즈
        net.eval()
        count = 0
        ## net.train(), net.eval()은 모델 클래스를 정의할 때 사용하는 부모 클래스인 nn.Module에서 정의되어 있다.

        for inputs, labels in test_loader:
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 예측 계산
            outputs = net(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 예측 라벨 산출
            predicted = torch.max(outputs, 1)[1]

            # 정답 건수 산출
            val_acc += (predicted == labels).sum().item()

            # 손실과 정확도 계산
            avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count

        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history
```

`net.train()`, `net.eval()`함수는 모델 클래스를 정의할 때 사용하는 부모 클래스인  `nn.Module`에서 정의되어 있다.

`nn.Dropout` , `nn.BatchNorm2d` 와 같은 레이어 함수에서는 각 함수에 대해 훈련 페이즈와 예측 페이즈를 구별하는 처리를 해줘야 한다.

### evaluate_history(학습 로그)

```python
# 학습 로그 해석

def evaluate_history(history):
    # 손실과 정확도 확인
    print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}')
    print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 학습 곡선 출력(손실)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='훈련')
    plt.plot(history[:,0], history[:,3], 'k', label='검증')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('손실')
    plt.title('학습 곡선(손실)')
    plt.legend()
    plt.show()

    # 학습 곡선 출력(정확도)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='훈련')
    plt.plot(history[:,0], history[:,4], 'k', label='검증')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('정확도')
    plt.title('학습 곡선(정확도)')
    plt.legend()
```

### show_images_labels(예측 결과 표시)

- loader : 검증용 데이터로더
- classes : 정답 데이터에 대응하는 라벨 값의 리스트 (ex. plane, car, bird ..)
- net : 사전에 학습이 끝난 모델의 인스턴스. None을 넘기면 정답 데이터만 표시되며, 학습 전에 데이터의 형태를 확인하고 싶은 경우에 사용한다.
- device : 예측 계산에 사용하는 디바이스

```python
# 이미지와 라벨 표시
def show_images_labels(loader, classes, net, device):

    # 데이터로더에서 처음 1세트를 가져오기
    for images, labels in loader:
        break
    # 표시 수는 50개
    n_size = min(len(images), 50)

    if net is not None:
      # 디바이스 할당
      inputs = images.to(device)
      labels = labels.to(device)

      # 예측 계산
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]
      #images = images.to('cpu')

    # 처음 n_size개 표시
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # net이 None이 아닌 경우는 예측 결과도 타이틀에 표시함
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 정답인지 아닌지 색으로 구분함
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # net이 None인 경우는 정답 라벨만 표시
        else:
          ax.set_title(label_name, fontsize=20)
        # 텐서를 넘파이로 변환
        image_np = images[i].numpy().copy()
        # 축의 순서 변경 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 값의 범위를[-1, 1] -> [0, 1]로 되돌림
        img = (img + 1)/2
        # 결과 표시
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()
```

### FNN vs CNN

```python
# FNN
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 은닉층 정의(은닉층의 노드수 : n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 출력층의 정의
        self.l2 = nn.Linear(n_hidden, n_output)

        # ReLU 함수 정의
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3
```

```python
# CNN
class CNN(nn.Module):
  def __init__(self, n_output, n_hidden):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.conv2 = nn.Conv2d(32, 32, 3)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d((2,2))
    self.flatten = nn.Flatten()
    self.l1 = nn.Linear(6272, n_hidden) # 6272 : 시뮬레이션의 결과 이용 또는 머릿속으로 해당 요소 수 계산
    self.l2 = nn.Linear(n_hidden, n_output)

    self.features = nn.Sequential(
        self.conv1,
        self.relu,
        self.conv2,
        self.relu,
        self.maxpool)

    self.classifier = nn.Sequential(
       self.l1,
       self.relu,
       self.l2)

  def forward(self, x):
    x1 = self.features(x)
    x2 = self.flatten(x1)
    x3 = self.classifier(x2)
    return x3
```

> 해당 포스팅은 '차근차근 실습하며 배우는 파이토치 딥러닝 프로그래밍'의 Ch09을 참고하여 작성되었습니다.