# 추억 사진관

## Intro

![](Appendix/intro.png)
>Insight Face는 훈련과 배치 모두에 최적화된 얼굴 인식, 얼굴 감지 및 얼굴 정렬과 같은 다양한 최첨단 알고리즘을 효율적으로 구현합니다.
>
>AWS에 서버를 두고 node.js와 monogoDB를 연결하며, v100서버에 flask를 통해 모델을 서빙하여 배포했습니다.




# Web Page URL
### http://memory-studio.ap-northeast-2.elasticbeanstalk.com/

## Member

![](Appendix/member.png)

|  이름  | github                          |
| :----: |  ------------------------------- |
| 김우진 |  https://github.com/w-jnn        |
| 신건희 |  https://github.com/Rigel0718    |
| 신중현 |  https://github.com/Blackeyes0u0 |
| 이종휘 |  https://github.com/gndldl       |


```
Face Recognition
├── Backbone
├── configs
├── docs
  ├── dataset.py
  ├── train.py
  ├── validation.py
  └── inference.py
    
Service
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
    

```

# Face Detection & Recognition

requirements.txt: 필요한 파이썬 라이브러리를 기술한 파일입니다.
model.py : 모델 파이썬 파일입니다.

---
## Service Architecture

![](Appendix/servicear.png)

>Dockerfile: 도커 이미지를 빌드하기 위한 파일입니다.
main.py: fastapi를 docker-compose 파일입니다.



![](Appendix/service.png)