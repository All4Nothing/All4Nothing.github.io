---
title:  "Mini-Batch with DataLoader"
metadate: "hide"
date : 2023-10-31 18:00:00 +0900
categories: [ ML/DL ]
image: "/assets/images/mini-batch-with-dataloader.png" 
---  

## Mini-Batch with DataLoader DataLoader를 활용한 미니 배치 데이터 생성

```python
from torch.utils.data import DataLoader

# mini-batch size
batch_size = 500

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

print(len(train_loader)) # 학습 데이터 수 / batch_size = 60000 / 500

for images, labels in train_loader:
	break

print(images.shape)
print(labels.shape)
```

```
120
torch.Size([500, 784])
torch.Size([500])
```

```python
n_input = image.shape[0]
n_output = len(set(list(labels.data.numpy())))
n_hidden = 128

class Net(nn.Module):
	def __*init*_(self, n_input, n_output, n_hidden):
		super().__**init__**()
		self.11 = nn.Linear(n_input, n_hidden)
		self.12 = nn.Linear(n_hidden, n_output)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x1 = self.11(x)
		x2 = self.relu(x1)
		x3 = self.12(x2)
		return x3

torch.manual_seed(123)
torch.cuda.manual_seed(123)

net = Net(n_input, n_output, n_hidden)
net = net.to(device)

lr = 0.01
net = Net(n_input, n_output, n_hidden).to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 100
history = np.zeros((0,5))

for tqdm.notebook import tqdm

for epoch in range(num_epochs):
	train_acc, train_loss = 0, 0
	val_acc, val_loss = 0, 0
	n_train, n_test = 0,0

# 훈련 페이즈
for inputs, labels in tqdm(train_loader):
	n_train += len(labels)
	
	# GPU로 전송
	inputs = inputs.to(device)
	labels = labels.to(device)

	# 경사 초기화
	optimizer.zero_grad()

	# 예측 계산
	outputs = net(inputs)

	# loss 계산
	loss = criterion(outputs, labels)
	
	# gradient 계산
	loss.backward()
	
	# parameter 수정
	optimizer.step()

# 예측 페이즈
for inputs_test, labels_test in test_loader:
	n_test += len(labels_test)

	inputs_test = inputs_test.to(device)
	labels_test = labels_test.to(device)

	# 예측 계산
	outputs_test = net(inputs_test)
	
	# loss 계산
	loss_test = criterion(outputs_test, labels_test)

	# 예측 label 산출
	predicted_test = torch.max(outputs_test, 1)[1]
	
	# loss와 accuracy 확인
	val_loss += loss_test.item()
	val_acc += (predicted_test == labels.test).sum().item()

# loss와 accuracy 확인
print(f'초기상태 : 손실 : {history[0,3]:.5f} 정확도 : {history[0,4]:.5f}')
print(f'최종상태 : 손실 : {history[-1,3]:.5f} 정확도 : {history[-1,4]:.5f}')

# learning curve(loss) 출력
plt.plot(history[:,0], history[:,1], 'b', label='train')
plt.plot(history[:,0], history[:,3], 'k', label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('learning curve(loss)')
plt.legend()
plt.show()

# learning curve(accuracy) 출력
plt.plot(history[:,0], history[:,2], 'b', label='train')
plt.plot(history[:,0], history[:,4], 'k', label='validation')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.title('learning curve(accuracy)')
plt.legend()
plt.show()
```

> 해당 포스팅은 '차근차근 실습하며 배우는 파이토치 딥러닝 프로그래밍'의 Ch08을 참고하여 작성되었습니다.