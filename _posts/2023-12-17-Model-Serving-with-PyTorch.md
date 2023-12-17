---
title:  "Model Serving with PyTorch"
metadate: "hide"
date : 2023-12-17 18:00:00 +0900
categories: [ Concepts ]
image: "/assets/images/model-serving-with-pytorch.png" 
---  
## Model Serving with PyTorch

모델이 운영 시스템에 배포하는 것을 Productionization(운영 이관)이라고 한다.

훈련된 PyTorch 모델을 serving하여 input 데이터에 대해 예측을 반환하는 pipeline을 구축하여, 모델 서버에 이 pipeline을 배치할 수 있다.

### Model Saving and Loading

먼저 훈련된 모델을 재사용할 수 있게 모델을 저장한다.

다음과 같이 모델 객체 전체를 저장하는 방법이 있다. 다만 이 방법은 모델의 parameter뿐만 아니라 사용된 모델의 class, directory 구조까지 저장하기 때문에, 나중에 모델의 class 또는 directory 구조가 변경되면 로딩을 할 수 없다.

```python
torch.save(model, PATH_TO_MODEL)
model = torch.load(PATH_TO_MODEL)
```

더 나은 방법으로는 다음과 같이 모델의 parameter만 저장할 수 있다.

```python
torch.save(model.state_dict(), PATH_TO_MODEL)
model = Convnet()
model = torch.load(PATH_TO_MODEL)
```

이와 같이 저장할 경우, 모델을 loading 할 때, 먼저 빈 모델 객체를 하나 인스턴스화 한 뒤 모델의 parameter를 로딩한다.

일반적으로 모델 정의는 python script로 작성된다.

```python
# cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32
        self.fc2 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op
```

그 후 다음과 같이 모델을 불러올 수 있다.

```python
from cnn_model import ConvNet
model = Convnet()
```

### Build Model Server with Flask

다음과 같이 Flask로 모델 서버를 구축할 수 있다.

```python
# server.py
import os
import json
import numpy as np
from flask import Flask, request

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_model import ConvNet

model = ConvNet()
PATH_TO_MODEL = "./convnet.pth"
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location="cpu"))
model.eval()

def run_model(input_tensor):
    model_input = input_tensor.unsqueeze(0)
    with torch.no_grad():
        model_output = model(model_input)[0]
    model_prediction = model_output.detach().numpy().argmax()
    return model_prediction

def post_process(output):
    return str(output)

app = Flask(__name__)

@app.route("/test", methods=["POST"])
def test():
    data = request.files['data'].read()
    md = json.load(request.files['metadata'])
    input_array = np.frombuffer(data, dtype=np.float32)
    input_image_tensor = torch.from_numpy(input_array).view(md["dims"])
    output = run_model(input_image_tensor)
    final_output = post_process(output)
    return final_output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890)
```

test 함수에 추가된 decorator는, 누군가 `/test` end-point에 `POST` 요청을 할 때마다 이 함수를 실행하도록 Flask 앱에 지시한다.

`POST` 요청이 들어오면, data와 meta-data를 읽어, data를 numpy array로 변환 후 pytorch tensor로 casting한다. 그 후, meta-data에서 이미지 dimension을 읽어 tensor를 제구성한다.

그 후, 모델의 input에 넣어 output을 구하고, 이를 return 한다.

마지막으로 server.py 마지막에 두 줄의 코드를 추가해, flask 서버를 0.0.0.0(localhost)에 port 번호 8890으로 호스팅한다.

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890)
```

그 후 다음과 같이 서버를 실행할 수 있다.

```python
python server.py
```

### Make Request

다음과 같이 입력된 이미지를 preprocessing하여 server에 request를 보낸다. 이미지 shape에 대한 tensor 정보를 문자열화하여 meta-data로 전송하고, 이미지 데이터 array는 byte로 변환해 POST 요청을 한다.

```python
# make_request.py
import io
import json
import requests
from PIL import Image

from torchvision import transforms

image = Image.open("./digit_image.jpg")

def image_to_tensor(image):
    gray_image = transforms.functional.to_grayscale(image)
    resized_image = transforms.functional.resize(gray_image, (28, 28))
    input_image_tensor = transforms.functional.to_tensor(resized_image)
    input_image_tensor_norm = transforms.functional.normalize(input_image_tensor, (0.1302,), (0.3069,))
    return input_image_tensor_norm

image_tensor = image_to_tensor(image)

dimensions = io.StringIO(json.dumps({'dims': list(image_tensor.shape)}))
data = io.BytesIO(bytearray(image_tensor.numpy()))

r = requests.post('http://localhost:8890/test',
                  files={'metadata': dimensions, 'data' : data})

response = json.loads(r.content)

print("Predicted digit :", response)
```

`r` 은 flask 서버에 보내는 request에 대한 response를 받는다. 이 response에는 모델이 반환한 output이 포함된다.

다음과 같이 request python script를 실행할 수 있다.

```python
python make_request.py
```

Predicted digit : 2

위와 같이 모델을 배포할 경우 특정 라이브러리를 설치하고, 모델을 저장, 로딩, 데이터 읽어오기 등의 작업이 필요하다. 이러한 작업들을 모두 수동으로 처리하면 모델 서버 개발 속도가 늦어진다. 

## Docker 도커

로컬 환경이 아닌 새로운 시스템에서 작업을 다시 해야한다면, 라이브러리를 설치하고, 파일을 현재 directory로 옮기는 등의 종속성 관리를 해줘야 하는 비효율적인 작업이 필요하다. 또한 서로 다른 시스템에서 서로 다른 버전의 라이브러리를 설치하게 되면 오류가 발생할 확률이 높다. 

Docker를 사용하면 OS 전체(소프트웨어 라이브러리, 설정 파일, 데이터 파일)를 가상화하여, 소프트웨어를 컨테이너화 할 수 있다.

https://docs.docker.com/get-started/overview

Docker를 사용하면 docker 이미지 형태로 dockerfile을 생성할 수 있다. Dockerfile을 사용하면 사전 설치된 python 라이브러리 또는 사용 가능한 모델에 대해 사전 작업 없이 빈 시스템에 build할 수 있다. 다른 시스템에도 일관되게 반복적으로 사용할 수 있는 일종의 OS 수준의 설계도를 만든다고 보면 된다. 

Dockerfile의 사용 예시는 다음과 같다.

먼저 필요한 python 라이브러리 목록을 만든다.

```docker
# requirements.txt
torch==1.5.0
torchvision==0.6.0
Pillow==6.2.2
Flask==1.1.1
```

그 후 dockerfile을 작성한다.

```docker
# Dockerfile
FROM python:3.8-slim

RUN apt-get -q update && apt-get -q install -y wget # wget 명령어 설치

COPY ./server.py ./ # local 개발 환경에 있는 파일을 가상 환경으로 복사
COPY ./requirements.txt ./

RUN wget -q https://raw.githubusercontent.com/All4Nothing/pytorch-DL-project/Ch10/convnet.pth
RUN wget -q https://github.com/wikibook/All4Nothing/raw/main/Ch10/digit_image.jpg

RUN pip install --no-cache-dir -r requirements.txt

USER root # docker client에 root 권한 부여
ENTRYPOINT ["python", "server.py"] # 이전 단계 모두 수행 후 'python server.py' 명렁어 실행
```

다음과 같이 dockerfile을 사용해 docker image를 build한다.

```bash
docker build -t digit_recognizer . # docker 이미지에 digit_recognizer 태그 할당
```

그 후 생성된 `digit_recognizer` 라는 이름의 docker image를 로컬 시스템에 배포하기 위해 다음 명령어를 실행한다. 명령어를 실행하면 `digit_recognizer` 도커 이미지를 사용해 시스템 내부의 가상 머신을 실행한다.

```bash
docker run -p 8890:8890 digit_recognizer
```

그 후 request를 보낸다.

```bash
pyhton make_request.py
```

런칭한 docker instance는 `Ctrl + C` 로 종료할 수 있으며, 다음과 같이 instance 및 도커 이미지를 삭제할 수 있다.

```bash
docker rm $(docker ps -a -q | head -1)
docker rmi $(docker images -q "digit_recognizer")
```

### +

이 외에도 torchserve를 통해 간단히 모델을 서빙할 수 있고 https://github.com/pytorch/serve/blob/master/README.md

torchscript와 ONNX를 이용해 pytorch 모델이 범용적으로 활용될 수 있게 모델을 내보낼 수 있다. https://pytorch.org/serve/configuration.html           https://pytorch.org/serve/server.html#advanced-features

또한 AWS, Google Cloud, Azer 클라우드 플랫폼에서 pytorch 모델을 훈련시키고 서빙할 수 있다.