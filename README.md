# 추억 사진관

## Intro

>사진을 찍고 나서 누구와 찍었는지 그 사람의 사진을 보고 싶을 때가 있다. 사진첩의 수 많은 사진들이 있는경우가 허다하기에, 찾기 어려울 수 있다. 우리가 만든 모델은 찾고 싶은 사람의 사진을 넣어주면 사진첩에서 그 사람을 찾아주는 프로젝트를 진행했다.


|  이름  | 역할 | github                          |
| :----: | ---- | ------------------------------- |
| 김우진 | 프론트&벡엔드     | https://github.com/w-jnn        |
| 신건희 | train 파이프라인 구현     | https://github.com/Rigel0718    |
| 신중현 | pretrained model 적용     | https://github.com/Blackeyes0u0 |
| 이종휘 | domain adaptation finetuning     | https://github.com/gndldl       |


```
Docker FastAPI baseline
├── Dockerfile
├── Docker-compose.yml
├── requirements.txt
└── app
	├── main.py
	├── model.py
	├── dataset.py
	└── inference.py

└── input_images
    └── image.jpg
    
    + model.pt + input_url + @
```

## Project Architecture

Dockerfile: 도커 이미지를 빌드하기 위한 파일입니다.
main.py: fastapi를 docker-compose로 불렁파일입니다.
model.py : 모델 파이썬 파일입니다.
requirements.txt: 필요한 파이썬 라이브러리를 기술한 파일입니다.
input_images: 모델에 입력으로 사용할 이미지가 들어 있는 폴더입니다

---

### Dockerfile 만들기

python3.10 기준으로 만들기 위해서 3.10 이미지를 사용하고 아래 두줄은 만약 도커로 배포를 한다면 주석해제하고 사용하면 된다.

도커의 역할은 3.10 파이썬 이미지로 필요한 라이브러리를 설치하는 것!

> 필요한 라이브러리 freeze 가져오기
```
pip freeze > requirements.txt
```

기본이 되는 docker image 를 고른다.
```Dockerfile
# 기반 이미지 설정 
FROM python:3.10.11
FROM pytorch/pytorch:latest

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .
COPY model.py .


COPY ./requirements.txt /code/requirements.txt

# 필요한 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 컨테이너 실행 시 모델 실행
CMD ["python", "model.py", "--input", "/app/input_images/image.jpg"]

RUN pip install --no-cache-dir -r /code/requirements.txt

#COPY ./app /code/app
 
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```
---
## Docker image 만들기
```bash
docker build -f Dockerfile -t blackeyes0u0/hi-ai:latest .
```
-f는 가져오려는 도커 flag이고, -t는 내가 만드려는 image이다.
blackeyes0u0는 내 username이고, hi-ai는 docker image 이름이다. latest는 이미지 태그이다. 

>>지금 문제는 도커 이미지에 cv2가 오류가 발생해서 그걸 해결하는 방법을 찾고 있다. 왜 문제인지 모르겠어서 
>
방법1. 현재 이미지에 조금씩 옵션을 바꿔주면서 고치기 
방법2. freeze 된 모든 라이브러리 설정대로 다운로드
방법3. opencv, pytorch 의 image를 기반으로 다운로드


#### docker-compose up
```docerk-compose.yml
version: "3.10.11"

services:
  fastapi:
    image: blackeyes0u0/hi-ai:new_version
    command: uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
    ports:
      - 8080:8080
    volumes:
      - ./app:/code/app
```
설정이 비교적 많지않다. 원하는 포트 설정을 해주면되고 uvicorn에서는 파일 변경이 일어나면 재시작을 하기 위해 –reload를 넣어준다.

volumns에서는 app폴더에서 수정이 이뤄지면 컨테이너 code/app에도 연결되어 반영되게하기 위해 ./app:/code/app으로 설정한다.

설정은 -끝-

### 사용 시작!
